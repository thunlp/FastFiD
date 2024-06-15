import logging
import os
import sys
from IndexBuilder import IndexBuilder
from arguments import parse_args
from utils import distributed_init, logger_config


if __name__ == "__main__":
    args = parse_args()
    logger_config(args)
    distributed_init(args)
    indexbuilder = IndexBuilder(args)
    indexbuilder.build_and_save_index()
