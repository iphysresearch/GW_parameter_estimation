export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


python ../model/main.py train new \
    --events_dir ../data/events_hdf5 \
    --model_dir ../output/models/demo1_model3 \
    --prior_dir demo.prior \
    --save_model_name model.pt \
    --waveform.sampling_frequency 4096 \
    --waveform.duration 8 \
    --waveform.conversion BBH \
    --waveform.waveform_approximant IMRPhenomPv2 \
    --waveform.reference_frequency 50 \
    --waveform.minimum_frequency 20 \
    --waveform.base bilby \
    --waveform.detectors H1 L1 \
    --waveform.target_time 1126259462.3999023 \
    --waveform.buffer_time 2 \
    --waveform.patch_size 0.5 \
    --waveform.overlap 0.5 \
    --waveform.stimulated_whiten \
    --waveform.norm_params_kind minmax \
    --waveform.target_optimal_snr 0 18.6 \
    --train.epoch_size 200 \
    --train.batch_size 2 \
    --train.num_workers 0 \
    --train.total_epochs 50000 \
    --train.lr_flow 0.00001 \
    --train.lr_embedding 0.00001 \
    --train.lr_anneal_method cosine \
    --train.output_freq 5 \
    --train.no_lr_annealing \
    --events.batch_size 16 \
    --events.nsamples_target_event 1000 \
    --events.event GW150914 \
    --events.flow 50 \
    --events.fhigh 250 \
    --events.sample_rate 4096 \
    --events.start_time 1126259456.3999023 \
    --events.duration 8 \
    --events.bilby_dir ../downsampled_posterior_samples_v1.0.0/ \
    --num_flow_steps 2 \
    umnn \
    --umnn_model.integrand_net_layers 50 50 50 \
    --umnn_model.cond_size 20 \
    --umnn_model.nb_steps 20 \
    --umnn_model.solver CCParallel \
    transformer \
    --transformer_cond.hidden_features 32 \
    --transformer_cond.num_blocks 2 \
    --transformer_cond.ffn_num_hiddens 16 \
    --transformer_cond.num_heads 2 \
    --transformer_cond.num_layers 2 \
    --transformer_cond.dropout 0.1 \
    --transformer_embedding.ffn_num_hiddens 512 \
    --transformer_embedding.num_heads 8 \
    --transformer_embedding.num_layers 6 \
    --transformer_embedding.dropout 0.1

# python ../model/main.py train existing \
#     --events_dir ../data/events_hdf5 \
#     --model_dir ../output/models/test_model \
#     --prior_dir demo.prior \
#     --save_model_name model.pt \
#     --waveform.sampling_frequency 4096 \
#     --waveform.duration 8 \
#     --waveform.conversion BBH \
#     --waveform.waveform_approximant IMRPhenomPv2 \
#     --waveform.reference_frequency 50 \
#     --waveform.minimum_frequency 20 \
#     --waveform.base bilby \
#     --waveform.detectors H1 L1 \
#     --waveform.target_time 1126259462.3999023 \
#     --waveform.buffer_time 2 \
#     --waveform.patch_size 0.5 \
#     --waveform.overlap 0.5 \
#     --waveform.stimulated_whiten \
#     --waveform.norm_params_kind minmax \
#     --waveform.target_optimal_snr 0 30 \
#     --train.epoch_size 50 \
#     --train.batch_size 8 \
#     --train.num_workers 0 \
#     --train.total_epochs 3 \
#     --train.no_lr_annealing \
#     --train.lr_flow 0.00001 \
#     --train.lr_embedding 0.00001 \
#     --train.lr_anneal_method cosine \
#     --train.output_freq 5 \
#     --events.batch_size 4 \
#     --events.nsamples_target_event 200 \
#     --events.event GW150914 \
#     --events.flow 50 \
#     --events.fhigh 250 \
#     --events.sample_rate 4096 \
#     --events.start_time 1126259456.3999023 \
#     --events.duration 8 \
#     --events.bilby_dir /Users/herb/Github/VisibleGWevents/Bilby-GWTC-1-Analysis-and-Verification/gwtc-1_analysis_results/downsampled_posterior_samples_v1.0.0/ \

#     # --train.no_lr_annealing \
#     # --prior_dir /Users/herb/Github/GW_parameter_estimation/GWToolkit/gwtoolkit/gw/prior_files/default.prior \    

