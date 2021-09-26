from torch.nn.modules import module
from argsparser import args_main
from main import PosteriorModel, transforms, Patching_data, ToTensor, Normalize_params, nn
from utils import print_dict
from torch import nn
import torch


def main():
    args = args_main()
    print_dict(args.__dict__, ncol=8, prex='\t')
    print_dict(args.run.__dict__, ncol=8, prex='\t')
    input()

    if args.run.name == 'train':
        print('Events directory\t', args.run.events_dir)
        print('Model directory\t\t', args.run.model_dir)
        pm = PosteriorModel(model_dir=args.run.model_dir,
                            events_dir=args.run.events_dir,
                            save_model_name=args.run.save_model_name,
                            use_cuda=args.cuda)
        print(f'Save the model as\t{args.run.save_model_name}')
        print('Device\t\t\t', pm.device)

        print('Init Waveform Dataset...')
        print_dict(vars(args.run.waveform), 5, '\t')
        waveform_arguments = dict(
            waveform_approximant=args.run.waveform.waveform_approximant,
            reference_frequency=args.run.waveform.reference_frequency,
            minimum_frequency=args.run.waveform.minimum_frequency)

        # ###################################################################################################
        # ###################################################################################################
        # Init pm.wfd and pm.wfd.load_prior_source_detector
        pm.init_WaveformDataset(args.run.waveform.sampling_frequency, args.run.waveform.duration,
                                args.run.waveform.conversion,
                                args.run.waveform.base, args.run.waveform.detectors, waveform_arguments,
                                filename=args.run.prior_dir)
        # if args.run.prior_dir is not None:
        #     print('Using priors in', args.run.prior_dir)
        # ###################################################################################################
        # ###################################################################################################
        # Init WaveformDatasetTorch & DataLoader + pm.input_shape/pm.target_labels ##########################
        # Hyper parameters start #############################
        print('Init Waveform PyTorch Dataset...')
        target_labels = ['mass_ratio', 'chirp_mass',
                         'luminosity_distance',
                         'dec', 'ra', 'theta_jn', 'psi', 'phase',
                         'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                         'geocent_time']
        start_time = args.run.waveform.target_time-(args.run.waveform.duration - args.run.waveform.buffer_time)
        geocent_time = (args.run.waveform.target_time-0.1, args.run.waveform.target_time+0.1)
        # Hyper parameters end ###############################
        composed_data = transforms.Compose([
            Patching_data(patch_size=args.run.waveform.patch_size,
                          overlap=args.run.waveform.overlap,
                          sampling_frequency=args.run.waveform.sampling_frequency),
            # output: (b c h)
        ])
        rand_transform_data = transforms.Compose([
            ToTensor(),
        ])
        composed_params = transforms.Compose([
            Normalize_params(args.run.waveform.norm_params_kind,
                             wfd=pm.wfd, labels=target_labels,
                             feature_range=(-1, 1)),
            ToTensor(),
        ])
        pm.init_WaveformDatasetTorch(
            args.run.optim.epoch_size,
            start_time,
            geocent_time,
            tuple(
                float(value) if i else int(value)
                for i, value in enumerate(args.run.waveform.target_optimal_snr)
            ),
            target_labels,
            args.run.waveform.stimulated_whiten,
            composed_data,
            composed_params,
            rand_transform_data,
            args.run.optim.batch_size,
            args.run.optim.num_workers,
            classification_ornot=True,
        )

        if args.run.existing is False:
            # ###################################################################################################
            # ###################################################################################################
            # Init embedding network ############################################################################
            print('Init Embedding Network...')
            embedding_transformer_kwargs = dict(
                isrel_pos_encoding=args.run.transformer.isrel_pos_encoding,
                ispso_encoding=args.run.transformer.ispso_encoding,
                vocab_size=0,  # 0 for embeding only
                ffn_num_hiddens=args.run.transformer.ffn_num_hiddens,
                num_heads=args.run.transformer.num_heads,
                num_layers=args.run.transformer.num_layers,
                dropout=args.run.transformer.dropout,
                valid_lens=None,
            )
            module_dict = {
                "3ex4": dict(
                    func=pm.init_rearrange,
                    kwargs=dict(pattern='b c h -> b c 1 h'),
                ),
                "3to4": dict(
                    func=pm.init_rearrange,
                    kwargs=dict(pattern='b (c t) h -> b c t h', c=len(args.run.waveform.detectors)),
                ),
                "vgg4to4": dict(
                    func=pm.init_vggblock,
                    kwargs=dict(),
                ),
                "4to3": dict(
                    func=pm.init_rearrange,
                    kwargs=dict(pattern='b c h w -> b (c h) w'),
                ),
                "transformer3to3": dict(
                    func=pm.init_vanilla_transformer,
                    kwargs=embedding_transformer_kwargs,
                ),
                "conformer3to3": dict(
                    func=pm.init_conformer,
                    kwargs=dict(),
                ),
                "cvt4to3": dict(
                    func=pm.init_cvt,
                    kwargs=dict(),
                ),
                "gap3to2": dict(
                    func=pm.init_global_average_pooling,
                    kwargs=dict(),
                ),
                "classifier2to2": dict(
                    func=pm.init_classifier,
                    kwargs=dict(),
                )
            }
            embedding_net = nn.Sequential()
            # embedding_modules = ["transformer3to3"]
            embedding_modules = ["3to4", "vgg4to4", "4to3", "transformer3to3", "gap3to2", "classifier2to2"]
            # embedding_modules = ["3ex4", "vgg4to4", "4to3", "transformer3to3", "conformer3to3"]
            # embedding_modules = ["conformer3to3"]
            # embedding_modules = ["3ex4", "cvt4to3"]
            # embedding_modules = ["3to4", "cvt4to3"]
            for name in embedding_modules:
                embedding_net.add_module(name, module_dict[name]['func'](**module_dict[name]['kwargs']))
            pm.init_embedding_network(embedding_net, args.run.optim)  # pn.embedding_net.to(pm.device)

        elif args.run.existing:
            pm.load_model()  # TODO
        # #######################################################################################################
        # #######################################################################################################
        # Init training #########################################################################################
        pm.init_training(args.run.optim)
        print('\tArgumentations for training:')
        print_dict(vars(args.run.optim), 3, '\t\t')

        # Training ###################################################################
        print('\tInference events during training:')
        print_dict(vars(args.run.inference), 3, '\t\t')

        try:
            pm.train(args.run.optim.total_epochs, args.run.optim.output_freq, args.run.inference)
        except KeyboardInterrupt as e:
            print(e)
        finally:
            print('Finished!')


    # elif args.run.name == 'test':
    #     pass
    else:
        raise


if __name__ == "__main__":
    main()
