from torch.nn.modules import module
from argsparser import args_main
from main import PosteriorModel, transforms, Patching_data, ToTensor, Normalize_params, nn
from utils import print_dict


def main():
    args = args_main()
    print_dict(args.__dict__, ncol=8, prex='\t')
    print_dict(args.comd.__dict__, ncol=8, prex='\t')
    input()

    if args.comd.name == 'train':
        print('Events directory\t', args.comd.events_dir)
        print('Model directory\t\t', args.comd.model_dir)
        pm = PosteriorModel(model_dir=args.comd.model_dir,
                            events_dir=args.comd.events_dir,
                            save_model_name=args.comd.save_model_name,
                            use_cuda=args.cuda)
        print(f'Save the model as\t{args.comd.save_model_name}')
        print('Device\t\t\t', pm.device)

        print('Init Waveform Dataset...')
        print_dict(vars(args.comd.waveform), 5, '\t')
        waveform_arguments = dict(
            waveform_approximant=args.comd.waveform.waveform_approximant,
            reference_frequency=args.comd.waveform.reference_frequency,
            minimum_frequency=args.comd.waveform.minimum_frequency)

        # ###################################################################################################
        # ###################################################################################################
        # Init pm.wfd and pm.wfd.load_prior_source_detector
        pm.init_WaveformDataset(args.comd.waveform.sampling_frequency, args.comd.waveform.duration,
                                args.comd.waveform.conversion,
                                args.comd.waveform.base, args.comd.waveform.detectors, waveform_arguments,
                                filename=args.comd.prior_dir)
        # if args.comd.prior_dir is not None:
        #     print('Using priors in', args.comd.prior_dir)
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
        start_time = args.comd.waveform.target_time-(args.comd.waveform.duration - args.comd.waveform.buffer_time)
        geocent_time = (args.comd.waveform.target_time-0.1, args.comd.waveform.target_time+0.1)
        # Hyper parameters end ###############################
        composed_data = transforms.Compose([
            Patching_data(patch_size=args.comd.waveform.patch_size,
                          overlap=args.comd.waveform.overlap,
                          sampling_frequency=args.comd.waveform.sampling_frequency),
            # output: (b c h)
        ])
        rand_transform_data = transforms.Compose([
            ToTensor(),
        ])
        composed_params = transforms.Compose([
            Normalize_params(args.comd.waveform.norm_params_kind,
                             wfd=pm.wfd, labels=target_labels,
                             feature_range=(-1, 1)),
            ToTensor(),
        ])
        pm.init_WaveformDatasetTorch(
            args.comd.optim.epoch_size,
            start_time,
            geocent_time,
            tuple(
                float(value) if i else int(value)
                for i, value in enumerate(args.comd.waveform.target_optimal_snr)
            ),
            target_labels,
            args.comd.waveform.stimulated_whiten,
            composed_data,
            composed_params,
            rand_transform_data,
            args.comd.optim.batch_size,
            args.comd.optim.num_workers,
        )

        if args.comd.existing is False:
            # ###################################################################################################
            # ###################################################################################################
            # Init embedding network ############################################################################
            print('Init Embedding Network...')
            embedding_transformer_kwargs = dict(
                isrel_pos_encoding=True,
                ispso_encoding=False,
                vocab_size=0,  # 0 for embeding only
                ffn_num_hiddens=args.comd.transformer.ffn_num_hiddens,
                num_heads=args.comd.transformer.num_heads,
                num_layers=args.comd.transformer.num_layers,
                dropout=args.comd.transformer.dropout,
                valid_lens=None,
            )
            module_dict = {
                "3ex4": dict(
                    func=pm.init_rearrange,
                    kwargs=dict(pattern='b c h -> b c 1 h'),
                ),
                "3to4": dict(
                    func=pm.init_rearrange,
                    kwargs=dict(pattern='b (c t) h -> b c t h', c=len(args.comd.waveform.detectors)),
                ),
                "vgg": dict(
                    func=pm.init_vggblock,
                    kwargs=dict(),
                ),
                "4to3": dict(
                    func=pm.init_rearrange,
                    kwargs=dict(pattern='b c h w -> b (c h) w'),
                ),
                "transformer": dict(
                    func=pm.init_vanilla_transformer,
                    kwargs=embedding_transformer_kwargs,
                ),
            }
            embedding_net = nn.Sequential()
            embedding_modules = ["3to4", "vgg", "4to3", "transformer"]
            embedding_modules = ["3ex4", "vgg", "4to3", "transformer"]
            for name in embedding_modules:
                embedding_net.add_module(name, module_dict[name]['func'](**module_dict[name]['kwargs']))

            # embedding_net = nn.Sequential(  # TODO
            #     ConformerEncoder(input_dim=pm.input_shape[-1], device=pm.device),
            # )
            # embedding_net = nn.Sequential(
            #     pm.init_rearrange('b c h -> b c 1 h'),
            #     # pm.init_rearrange('b (c t) h -> b c t h', c=2),
            #     pm.init_cvt(),
            # )
            pm.init_embedding_network(embedding_net)  # pn.embedding_net.to(pm.device)
            input('111')
            # ###################################################################################################
            # ###################################################################################################
            # Init nflow network ################################################################################
            print('Init Normalizing Flow Network...')
            print(f'\tNumber of transforms in flow sequence: {args.num_flow_steps}')
            pm.get_base_transform_kwargs(args)
            pm.init_nflow_network(args.num_flow_steps)
        elif args.comd.existing:
            pm.load_model()

    # elif args.comd.name == 'test':
    #     pass
    else:
        raise


if __name__ == "__main__":
    main()
