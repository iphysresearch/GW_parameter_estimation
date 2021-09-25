from dataclasses import dataclass, field
from typing import Union, List, Literal
from pathlib import Path
from simple_parsing import ArgumentParser  # subparsers
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn
# https://github.com/lebrice/SimpleParsing


@dataclass
class ResNetConditioner:
    """HyperParameters for ResNet Conditioner
    """
    # name of the task
    name: str = "ResNetConditioner"
    # number of hidden neurals
    hidden_dims: int = 512
    # activation function (Literal['relu', 'leaky_relu', 'elu'])
    activation: str = 'relu'
    # dropout
    dropout: float = 0.0
    # num_blocks
    num_blocks: int = 2
    # batch_norm
    batch_norm: bool = True


@dataclass
class TransformerConditioner:
    """HyperParameters for Transformer Conditioner
    """
    # name of the task
    name: str = "TransformerConditioner"
    # hidden_features
    hidden_features: int = 4
    # number of hidden neurals in FFN
    ffn_num_hiddens: int = 16
    # number of heads for self-attention
    num_heads: int = 2
    # num_blocks
    num_blocks: int = 2
    # number of the transformer block
    num_layers: int = 2
    # dropout probability
    dropout: float = 0.1


@dataclass
class RQNSFCFlow:
    """HyperParameters for RQ-NSF(C)
    """
    conditioner: Union[ResNetConditioner, TransformerConditioner]

    # name of the task
    name: str = "RQ-NSF(C)"
    # number of bins for Spline func.
    num_bins: int = 8
    # tail_bound
    tail_bound: float = 1.0
    # apply_unconditional_transform
    apply_unconditional_transform: bool = True


@dataclass
class UMNNFlow:
    """HyperParameters for UMNN
    An unconstrained monotonic neural networks coupling layer that transforms the variables.
    """
    conditioner: Union[ResNetConditioner, TransformerConditioner]

    # name of the task
    name: str = 'UMNN'
    # The layers dimension to put in the integrand network.
    integrand_net_layers: List[int] = field(default_factory=lambda: [50, 50, 50])
    # The embedding size for the conditioning factors.
    cond_size: float = 20
    # The number of integration steps.
    nb_steps: float = 20
    # The quadrature algorithm - CC or CCParallel. Both implements Clenshaw-Curtis quadrature with'
    # Leibniz rule for backward computation. CCParallel pass all the evaluation points (nb_steps) at once, it is faster
    # but requires more memory.
    solver: str = 'CCParallel'


@dataclass
class VanillaTransformerParameters:
    """HyperParameters for a WaveformDataset
    """
    # name of the task
    name: str
    # Use Relative Global Attention or not
    isrel_pos_encoding: bool = True
    # Use Positional Encoding or not
    ispso_encoding: bool = False
    # size of the dictionary of embeddings, 0 for no embeddings
    vocab_size: int = 0
    # number of hidden neurals in FFN
    ffn_num_hiddens: int = 128
    # number of heads for self-attention
    num_heads: int = 2
    # number of the transformer block
    num_layers: int = 6
    # dropout probability
    dropout: float = 0.1
    # valid length for masking
    valid_lens: Union[list, None] = None
    # normalized shape of input data (need to be pre-defined)
    norm_shape: Union[list, None] = None
    # key_size (need to be pre-defined)
    key_size: Union[int, None] = None
    # query_size (need to be pre-defined)
    query_size: Union[int, None] = None
    # value_size (need to be pre-defined)
    value_size: Union[int, None] = None
    # numbers features of input data (need to be pre-defined)
    num_hiddens: Union[int, None] = None
    # ffn_num_input (need to be pre-defined)
    ffn_num_input: Union[int, None] = None


@dataclass
class WaveformDatasetParameters:
    """HyperParameters for a WaveformDataset
    """
    # name of the task
    # name: str
    # sampling frequency
    sampling_frequency: int = 4096
    # duration of the waveform strain (sec)
    duration: float = 8
    # conversion for 'BBH' or 'BNS'
    conversion: str = 'BBH'
    # waveform family ('IMRPhenomPv2', 'SEOBNRv4P')
    waveform_approximant: str = 'IMRPhenomPv2'
    # reference frequency of the waveform
    reference_frequency: float = 50
    # minimum frequency of the waveform
    minimum_frequency: float = 20
    # waveform generator ('bilby', 'pycbc')
    base: str = 'bilby'
    # detectors for waveform (A1 H1 L1 V1 K1 CE ET GEO600)
    detectors: List[str] = field(default_factory=lambda: ['H1', 'L1'])
    # detector responce at target time GPS
    target_time: float = 1126259462.3999023
    # waveform buffer time from target time to the end
    buffer_time: float = 2
    # patching waveforms to target input size (perc)
    patch_size: float = 0.5
    # patching waveforms with overlap (perc)
    overlap: float = 0.5
    # whiten of not based on stimulation
    stimulated_whiten: bool = True
    # the way for normalizing parameters
    norm_params_kind: str = 'minmax'
    # normalizing waveforms on (detector, optimal_snr) or None
    target_optimal_snr: Union[tuple, None] = (0, 18.6)


@dataclass
class OptimizationParameters:
    """HyperParameters for Optimization
    """
    # name of the task
    # name: str
    # number of strains in a waveform block
    epoch_size: int = 128
    # batch size
    batch_size: int = 16
    # num of workers for DataLoader
    num_workers: int = 0
    # number of epochs for training
    total_epochs: int = 10_000
    # learning rates for flow model
    lr_flow: float = 0.000_1
    # learning rates for embedding model
    lr_embedding: float = 0.000_1
    # annealing learning rate or not
    lr_annealing: bool = True
    # anneal method for lr ('step', 'cosine', 'cosineWR')
    lr_anneal_method: str = 'cosine'
    # gamma for step anneal method
    steplr_gamma: float = 0.5
    # step size for step anneal method
    steplr_step_size: int = 80
    # print output for the number of batch size during training
    output_freq: int = 50
    # save the model or not
    save: bool = True
    # pretrain_embedding_dir
    pretrain_embedding_dir: Path = Path("./output/model/")


@dataclass
class InferenceEventsParameters:
    """HyperParameters for Inference GW events
    """
    # name of the task
    # name: str
    batch_size: int = 4
    nsamples_target_event: int = 200
    event: str = 'GW150914'
    flow: float = 50
    fhigh: float = 250
    sample_rate: int = 4096
    start_time: float = 1126259456.3999023
    duration: float = 8
    bilby_dir: Path = Path("../downsampled_posterior_samples_v1.0.0/")


@dataclass
class Train(Serializable):
    """Example of a command to start a Training run."""
    # Choose a normalizing flow model
    flowmodel: Union[RQNSFCFlow, UMNNFlow]

    # target events data directory
    events_dir: Path = Path("~/trainNew")
    # model directory
    model_dir: Path = Path("./")
    # target model file name
    save_model_name: str = "model.pt"
    # prior file directory
    prior_dir: Path = Path("./demo.prior")
    # is existing ?
    existing: bool = False

    waveform: WaveformDatasetParameters = WaveformDatasetParameters(
    )

    optim: OptimizationParameters = OptimizationParameters(
    )

    inference: InferenceEventsParameters = InferenceEventsParameters(
    )

    transformer: VanillaTransformerParameters = VanillaTransformerParameters(
        "vanilla"
    )

    # num of flow steps
    num_flow_steps: int = 2

    def saveyaml(self):
        if self.existing:
            print(f"Loading in directory {self.model_dir}")
            self = self.load(self.model_dir / "args.yaml", drop_extra_fields=False)
        # print(f"Saving in directory {self.model_dir}, {type(self.model_dir)}")
        self.save(self.model_dir / "args.yaml")
        return self

    @property
    def name(self):
        return 'train'


@dataclass
class Test:
    """Example of a command to start a Test run."""
    # the testing directory
    test_dir: Path = Path("~/test")

    def execute(self):
        print(f"Testing in directory {self.test_dir}")

    @property
    def name(self):
        return 'test'


@dataclass
class Program(Serializable):
    """Some top-level command"""
    run: Union[Train, Test]
    # us cuda or not
    cuda: bool = True
    # log additional messages in the console.
    verbose: bool = False

    def execute(self):
        print(f"Executing Program (verbose: {self.verbose})")
        # self.run.execute()
        self.run = self.run.saveyaml()


@encode.register
def encode_path(obj: Path) -> str:
    """ We choose to encode a Path as a str, for instance """
    return str(obj)


def args_main(debug=False):

    register_decoding_fn(Path, Path)
    register_decoding_fn(tuple, tuple)
    register_decoding_fn(list, list)

    parser = ArgumentParser()
    parser.add_arguments(Program, dest="prog")
    if debug:
        return parser.parse_args()
    args = parser.parse_args()
    prog: Program = args.prog
    prog.execute()
    return prog


if __name__ == "__main__":
    args = args_main()
    prog: Program = args.prog

    print("prog:", prog)

    prog.execute()

    print("prog:", prog)
