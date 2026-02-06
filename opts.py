import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'coefficients': [
            dict(name='--lambda_0',
                 default='0.5',
                 type=float,
                 help='Penalty Coefficient that Controls the Penalty Extent in PCCE'),
        ],
        'paths': [
            dict(name='--model_pretrained',
                 default='pretrained-models/kinetics_squeezenet_RGB_16_best.pth',
                 type=str,
                 help='Global path of pretrained 3d resnet101 model (.pth)'),
            dict(name='--network_choose',
                 default='squeezenet',
                 help='network_choose'
                 ),
            dict(name='--root_path',
                 default="BrainGuided",
                 type=str,
                 help='Global path of root directory'),
            dict(name='--data_root_path',
                 default="BrainGuided",
                 type=str,
                 help='Global path of root directory'),
            dict(name="--video_path",
                 default="EK6--imgs",
                 type=str,
                 help='Local path of videos', ),
            dict(name="--video_raw_path",
                 default="EK6--raw",
                 type=str,
                 help='Local path of videos', ),
            dict(name="--annotation_path",
                 default='video_id_ek6.csv',
                 type=str,
                 help='Local path of annotation file'),
            dict(name='--dataset_choose',
                 default='rt',
               #   default='ek6',
                 help='dataset_choose'
                 ),
            dict(name='--behavior',
                 default=False,
                 ),
            dict(name='--behavior_data',
                 default='semantic',
                 ),
            dict(name='--n_classes',
                 default=6,
                 type=int,
                 help='Number of classes'),
            dict(name="--neural_video_path",
                 default="nRT--imgs/nRT",
               #   default="iScience--imgs/iScience",
                 type=str,
                 help='Local path of neural videos', ),
            dict(name="--neural_video_raw_path",
                 default="nRT--raw/nRT",
               #   default="iScience--raw/iScience",
                 type=str,
                 help='Local path of neural videos', ),
            dict(name="--result_path",
                 default='results',
                 type=str,
                 help="Local path of result directory"),
            dict(name='--expr_name',
                 type=str,
                 default=''),
            dict(name="--task",
                 default="design",
                 type=str,
                 help="classification tasks e.g. design, space, ..."),
        ],
        'core': [
            dict(name='--batch_size',
                 default=16,
                 type=int,
                 help='Batch Size'),
            dict(name='--batch_size_neural',
                 default=16,
                 type=int,
                 help='Batch Size Neural'),
            dict(name='--batch_size_behavior',
                 default=32,
                 type=int,
                 help='Batch Size Behavior'),
            dict(name='--snippet_duration',
                 default=16,
                 type=int),
            dict(name='--sample_size',
                 default=112,
                 type=int,
                 help='Heights and width of inputs'),
            dict(name='--seq_len',
                 default=1,
               #   default=10,
                 type=int),
            dict(name='--loss_func',
                 default='ce',
                 type=str,
                 help='ce | pcce_ve8'),
            dict(name='--learning_rate',
               #   default=1e-5,
                 default=2e-4,
                 type=float,
                 help='Initial learning rate', ),
            dict(name='--weight_decay',
                 default=0,
                 type=float,
                 help='Weight Decay'),
            dict(name='--fps',
                 default=30,
                 type=int,
                 help='fps'),
            dict(name='--split',
                 default=2,
                 type=int,
                 help='split'),
            dict(name='--use_model',
                 default=True,
                 help='use predicted response'
                 ),
            dict(name='--data_use',
                 default='sub-02',
               #   default='mean',
                 help='data use'
                 ),
            dict(name='--roi',
                 default='',
                 help='which roi to calculate layer contribution (e.g. EVC, RSC, PPA, TOS)'
                 ),
            dict(name='--RSA_similarity_print',
                 default=False,
                 help='data use'
                 ),
            dict(name='--alpha',
                 default=25,
                 help='alpha'
                 ),
            dict(name='--co_train',
                 default=True,
                 help='co_train'
                 ),
            dict(name='--voxel_select_threshold',
                 default=0.15,
                 help='dataset_choose'
                 ),
            dict(name='--sig_test_run',
                 default=1,
                 help='dataset_choose'
                 ),
            dict(name='--video_num',
                 default=744,
               #   default=800,
                 help='dataset_choose' # in neural data
                 ),
            dict(name='--random_choice',
                 default=False,
                 help='dataset_choose'
                 ),
            dict(name='--use_lstm',
                 default=False,
                 help='train lstm together when remain time info'
                 ),
            dict(name='--train_from_checkpoint',
                 default=False,
                 help='train_from_checkpoint'
                 ),
            dict(name='--align_only_last_layer',
                 default=False,
                 help='align_only_last_layer'
                 ),
            dict(name='--get_layer_contribution',
                 default=True,
                 help='calculate layer contribution'
                 ),
            dict(name='--contribution_method',
                 default='ridge',
                 type=str,
                 help='ridge | rdm_corr'
                 ),
            dict(name='--train_only_layer_contribution',
                 default=False,
                 help='train only layer contribution'
                 ),
            dict(name='--mixup_pct',
                 default=.0,
                 help='proportion of way through training when to switch from BiMixCo to mse'
                 ),
            dict(name='--freezeall',
                 default=False,
                 help='freeze cnn, unfreeze classifier'
                 ),
            dict(name='--freezehalf',
                 default=False,
                 help='freeze half cnn, unfreeze classifier'
                 ),
            dict(name='--add_mse',
                 default=False,
                 help='CE+CKA+MSE'
                 ),
            dict(name='--rho4rdm',
                 default=False,
                 help='use spearman rank correlation to calculate rdm'
                 ),
        ],
        'network': [
            {
                'name': '--audio_embed_size',
                'default': 256,
                'type': int,
            },
            {
                'name': '--audio_n_segments',
                'default': 16,
                'type': int,
            }
        ],

        'common': [
            dict(name='--use_cuda',
                 action='store_true',
                 default=False,
                 help='only cuda supported!'
                 ),
            dict(name='--debug',
                 default='True',
                 action='store_true'),
            dict(name='--dl',
                 action='store_true',
                 default=False,
                 help='drop last'),
            dict(
                name='--n_threads',
                default=4,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(
                name='--n_epochs',
               #  default=150,
                default=100,
                type=int,
                help='Number of total epochs to run',
            )
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args
