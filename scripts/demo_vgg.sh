export CUDA_VISIBLE_DEVICES=1

python model/run.py train \
--model_dir ./output/models/demo_model_pretrain \
--events_dir ./data/events_hdf5 \
--prior_dir scripts/demo.prior \
--bilby_dir ./downsampled_posterior_samples_v1.0.0 \
--num_flow_steps 15 \
--ffn_num_hiddens 128 \
--lr_flow 0.000001 \
--lr_embedding 0.0 \
--target_optimal_snr 0 0 \
--optim.batch_size 8 \
--epoch_size 512 \
--output_freq 5 \
--pretrain_embedding_dir ./output/models/demo_model_classification \
rqnsfcflow  transformerconditioner \
--ffn_num_hiddens 128 \
--num_blocks 2 \
--num_layers 4 \
--hidden_features 128
