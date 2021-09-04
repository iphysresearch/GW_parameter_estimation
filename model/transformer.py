import torch
from torch import nn
from torch.nn import functional as F
import math


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来执行 softmax 操作"""
    # `X`: 3D张量, `valid_lens`: 1D或2D 张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    # 在最后的轴上，被遮蔽的元素使用一个非常大的负值替换，从而其 softmax (指数)输出为 0
    X = sequence_mask(X.reshape(-1, shape[-1]),
                      valid_lens,
                      value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`, 查询的个数, 1, `num_hidden`)
        # `key` 的形状：(`batch_size`, 1, “键－值”对的个数, `num_hiddens`)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # `self.w_v` 仅有一个输出，因此从形状中移除最后那个维度。
        # `scores` 的形状：(`batch_size`, 查询的个数, “键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    # `queries` 的形状：(`batch_size`, 查询的个数, `d`)
    # `keys` 的形状：(`batch_size`, “键－值”对的个数, `d`)
    # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
    # `valid_lens` 的形状: (`batch_size`,) 或者 (`batch_size`, 查询的个数)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class RelativeGlobalAttention(nn.Module):
    """
    Ref: https://jaketae.github.io/study/relative-positional-encoding/
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout=0.1, bias=False, max_len=1024, **kwargs):
        super().__init__(**kwargs)
        assert key_size == query_size == value_size == num_hiddens
        d_model = key_size
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)
        self.attention = DotProductAttention(dropout)

    def forward(self, queries, keys, values, valid_lens):
        assert queries.shape == keys.shape == values.shape
        x = queries
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        if valid_lens:  # TODO  整合 RelativeGlobalAttention 到 MultiHeadAttention 里
            mask = self.mask[:, :, :seq_len, :seq_len]
            # mask.shape = (1, 1, seq_len, seq_len)
            attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        self.attention.attention_weights = out
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # `queries`, `keys`, or `values` 的形状:
        # (`batch_size`, 查询或者“键－值”对的个数, `num_hiddens`)
        # `valid_lens`　的形状:
        # (`batch_size`,) or (`batch_size`, 查询的个数)
        # 经过变换后，输出的 `queries`, `keys`, or `values`　的形状:
        # (`batch_size` * `num_heads`, 查询或者“键－值”对的个数,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴 0，将第一项（标量或者矢量）复制 `num_heads` 次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # `output` 的形状: (`batch_size` * `num_heads`, 查询的个数,
        # `num_hiddens` / `num_heads`)

        output = self.attention(queries, keys, values, valid_lens)

        # `output_concat` 的形状: (`batch_size`, 查询的个数, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    # 输入 `X` 的形状: (`batch_size`, 查询或者“键－值”对的个数, `num_hiddens`).
    # 输出 `X` 的形状: (`batch_size`, 查询或者“键－值”对的个数, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出 `X` 的形状: (`batch_size`, `num_heads`, 查询或者“键－值”对的个数,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # `output` 的形状: (`batch_size` * `num_heads`, 查询或者“键－值”对的个数,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 `transpose_qkv` 函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的 `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
                10000,
                torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
                num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, isrel_pos_encoding=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        if isrel_pos_encoding:
            self.attention = RelativeGlobalAttention(key_size, query_size,
                                                     value_size, num_hiddens,
                                                     num_heads, dropout, use_bias)
        else:
            self.attention = MultiHeadAttention(key_size, query_size,
                                                value_size, num_hiddens,
                                                num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, valid_lens,
                 use_bias=False, ispso_encoding=True, isrel_pos_encoding=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.valid_lens = valid_lens
        self.embedding = nn.Embedding(vocab_size, num_hiddens) if vocab_size else None
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout) if ispso_encoding else (lambda x: x)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias, isrel_pos_encoding))

    def forward(self, X, *args):
        # 因为位置编码值在 -1 和 1 之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding((X if self.embedding is None else self.embedding(X)) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, self.valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """解码器中第 i 个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size,
                                             value_size, num_hiddens,
                                             num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size,
                                             value_size, num_hiddens,
                                             num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有标记都在同一时间处理，
        # 因此 `state[2][self.i]` 初始化为 `None`。
        # 预测阶段，输出序列是通过标记一个接着一个解码的，
        # 因此 `state[2][self.i]` 包含着直到当前时间步第 `i` 个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # `dec_valid_lens` 的开头: (`batch_size`, `num_steps`),
            # 其中每一行是 [1, 2, ..., `num_steps`]
            dec_valid_lens = torch.arange(1, num_steps + 1,
                                          device=X.device).repeat(
                                              batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # `enc_outputs` 的开头: (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

    @property
    def attention_weights(self):
        raise NotImplementedError


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


####

class TransformerResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_features,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        context_tokens=None,
        zero_initialization=True,
    ):
        super().__init__()

        if context_tokens is not None:
            self.context_layer = nn.Conv1d(
                in_channels=context_tokens,
                out_channels=hidden_features,
                kernel_size=1,
                padding=0,
            )
        self.transformer_block = TransformerEncoder(
            valid_lens=None,
            vocab_size=0,  # for embeding only
            key_size=norm_shape[1],
            query_size=norm_shape[1],
            value_size=norm_shape[1],
            num_hiddens=norm_shape[1],
            norm_shape=norm_shape,
            ffn_num_input=norm_shape[1],
            ffn_num_hiddens=ffn_num_hiddens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            )
        # if zero_initialization:
        #     pass
        #  init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
        #  init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        temps = self.transformer_block(temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class TransformerResidualNet(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_tokens,
        num_blocks=2,
        ffn_num_hiddens=128,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        context_features=None,
    ):
        super().__init__()
        self.context_tokens = context_tokens
        self.context_features = context_features
        self.hidden_features = hidden_features
        if context_features is not None:
            self.initial_layer = nn.Conv1d(
                in_channels=in_features + context_tokens,
                out_channels=hidden_features,
                kernel_size=1,
                padding=0,
            )
        else:
            self.initial_layer = nn.Conv1d(
                in_channels=in_features,
                out_channels=hidden_features,
                kernel_size=1,
                padding=0,
            )
        self.blocks = nn.ModuleList(
            [
                TransformerResidualBlock(
                    hidden_features=hidden_features,
                    context_tokens=context_tokens,
                    norm_shape=(hidden_features, context_features),
                    ffn_num_hiddens=ffn_num_hiddens,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.flatten_layer = nn.Flatten()
        flatten_features = hidden_features * context_features
        self.final_layer = nn.Linear(flatten_features, out_features)

    def forward(self, inputs, context=None):
        inputs = inputs.unsqueeze(-1).repeat(1, 1, self.context_features)
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context)
        return self.final_layer(self.flatten_layer(temps))
