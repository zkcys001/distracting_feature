"""Entry point."""
import os

import torch

import config
import utils
import trainer, trainer_rl_typeloss
from data.load_data import load_data

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.gpunum > 0:
        torch.cuda.manual_seed(args.random_seed)
    if args.rl:

        trainer_rl_typeloss.main(args)

    else:
        trainer.main(args)


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
