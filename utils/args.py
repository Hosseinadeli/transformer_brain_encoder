from pathlib import Path
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="NSD Training", add_help=False)

    parser.add_argument("--resume", default=None, help="resume from checkpoint")
    parser.add_argument(
        "--output_path",
        default="./checkpoints",
        type=str,
        help="if not none, then store the model resuls",
    )

    parser.add_argument("--save_model", default=True, type=int)

    ## NSD params
    parser.add_argument("--subj", default=1, type=int)
    parser.add_argument("--run", default=1, type=int)
    parser.add_argument(
        "--data_dir",
        default="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
        type=str,
    )
    parser.add_argument(
        "--imgs_dir",
        default="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
        type=str,
    )
    parser.add_argument(
        "--parcel_dir",
        default="./parcels/checkpoints",
        type=str,
    )
    parser.add_argument(
        "--parent_submission_dir",
        default="./algonauts_2023_challenge_submission/",
        type=str,
    )

    parser.add_argument("--saved_feats", default=None, type=str)  #'dinov2q'
    parser.add_argument(
        "--saved_feats_dir", default="../../algonauts_image_features/", type=str
    )

    parser.add_argument(
        "--readout_res",
        choices=[
            "voxels",
            "streams_inc",
            "visuals",
            "bodies",
            "faces",
            "places",
            "words",
            "hemis",
            "parcels",
        ],  # TODO: add clusters
        default="parcels",
        type=str,
    )

    # the model for mapping from backbone image features to fMRI
    parser.add_argument(
        "--encoder_arch",
        choices=["transformer", "linear"],
        default="transformer",
        type=str,
    )

    parser.add_argument(
        "--objective",
        choices=["NSD"],
        default="classification",
        help="which model to train",
    )

    # Backbone
    parser.add_argument(
        "--backbone_arch",
        choices=[
            None,
            "dinov2",
            "dinov2_q",
            "resnet18",
            "resnet50",
            "dinov2_special_token",
            "dinov2_q_special_token",
        ],
        default="dinov2_q",
        type=str,
        help="Name of the backbone to use",
    )  # resnet50 resnet18 dinov2

    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--return_interm",
        default=False,
        help="Train segmentation head if the flag is provided",
    )

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=0,
        type=int,
        help="Number of encoding layers in the transformer brain model",
    )
    parser.add_argument(
        "--dec_layers",
        default=1,
        type=int,
        help="Number of decoding layers in the transformer brain model",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=768,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )  # 256  #868 (100+768)
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=16,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=16, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    parser.add_argument(
        "--enc_output_layer",
        default=1,
        type=int,
        help="Specify the encoder layer that provides the encoder output. default is the last layer",
    )

    # training parameters
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of data loading num_workers"
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size")
    parser.add_argument(
        "--lr", default=0.0005, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="weight decay "
    )
    parser.add_argument("--lr_drop", default=4, type=int)
    parser.add_argument("--lr_backbone", default=0, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument("--evaluate", action="store_true", help="just evaluate")

    parser.add_argument("--wandb_p", default="brain_encoder", type=str)
    parser.add_argument("--wandb_r", default=None, type=str)

    # dataset parameters
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help="what size should the image be resized to?",
    )
    parser.add_argument(
        "--horizontal_flip",
        default=True,
        help="whether to use horizontal flip augmentation",
    )

    parser.add_argument(
        "--img_channels",
        default=3,
        type=int,
        help="what should the image channels be (not what it is)?",
    )  # gray scale 1 / color 3

    parser.add_argument("--lh_vs", default=None)
    parser.add_argument("--rh_vs", default=None)

    parser.add_argument(
        "--axis", default="anterior", choices=["anterior", "posterior"], type=str
    )
    parser.add_argument("--hemi", default="lh", choices=["lh", "rh"], type=str)

    return parser


def get_default_args():
    parser = get_args_parser()
    default_args = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }
    args = argparse.Namespace(**default_args)
    return args


def get_model_dir_args(
    args,
):
    return get_model_dir(
        args.output_path,
        args.backbone_arch,
        args.encoder_arch,
        args.subj,
        args.enc_output_layer,
        args.run,
        args.hemi,
    )


def get_model_dir(
    output_path, backbone_arch, encoder_arch, subj, enc_output_layer, run, hemi
):
    p = (
        Path(output_path)
        / f"nsd_test/{backbone_arch}_{encoder_arch}"
        / f"subj_{int(subj):02}"
        / f"enc_{enc_output_layer}"
        / f"run_{run}"
        / hemi
    )

    return p


def get_run_dir(args):
    return get_model_dir_args(args) / ".."