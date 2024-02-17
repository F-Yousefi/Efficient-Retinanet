"""This is the main module in the project which gets users prefernces."""

from optparse import OptionParser
from utils.config import config
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = OptionParser()
    parser.add_option(
        "-p",
        "--path",
        dest="dir_to_dataset",
        help="Path to the desired dataset.",
        default=config.dir_to_dataset,
    )

    parser.add_option(
        "-d",
        "--device",
        dest="availabe_device",
        help="Choose the device you want to start training on.['cuda','cpu']",
        default=config.device,
    )

    parser.add_option(
        "-g",
        "--gpu",
        dest="GPU_under_control",
        help="True if you don't want to put a lot of pressure on your gpu card. It will keep your GPU's temperature in the safe zone.",
        default=config.gpu_under_control,
    )

    parser.add_option(
        "-m",
        "--model",
        dest="backbone_model_name",
        help="The possible backbone models are as follows: efficientnet-b[0-7] -> efficientnet-b4 ",
        default=config.backbone_model_name,
    )

    parser.add_option(
        "-l",
        "--load",
        dest="dir_to_pretrained_model",
        help="The directory to a pretrained model checkpoint.",
        default=config.dir_to_pretrained_model,
    )

    parser.add_option(
        "-w",
        "--workers",
        dest="num_workers",
        help="The directory to a pretrained model checkpoint.",
        default=config.num_workers,
    )

    parser.add_option(
        "-t",
        "--train",
        dest="train",
        help="If you want to start training your model based on your dataset, set this arg True. Otherwise, it just monitor the performance of your pretrained model.",
        default=True,
    )

    (options, args) = parser.parse_args()
    config.dir_to_dataset = options.dir_to_dataset
    config.device = options.availabe_device
    config.gpu_under_control = options.GPU_under_control
    config.dir_to_pretrained_model = options.dir_to_pretrained_model
    config.backbone_model_name = options.backbone_model_name
    config.num_workers = int(options.num_workers)
    return options.train


if __name__ == "__main__":
    train_mode = main()
    if train_mode:
        from trainer import lunch

        lunch()
    else:
        from monitor import lunch

        lunch()
