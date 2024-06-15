import argparse
from email.policy import default
import os

def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='UnitedQA Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_training_args(parser)
    parser = _add_dataset_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_reader_args(parser)
    parser = _add_retriever_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.data_parallel_size = args.world_size
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))
    # eval batch size
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    
    if args.global_batch_size is None and args.batch_size is not None:
        args.global_batch_size = args.batch_size * args.data_parallel_size
    if args.global_batch_size is not None and args.batch_size is not None:    
        args.gradient_accumulation_steps = args.global_batch_size // (args.batch_size * args.data_parallel_size)

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])
    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('-------------------- arguments --------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (32 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('---------------- end of arguments ----------------', flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--batch-size', type=int, default=None,
                       help='Train batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Evaluation batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Global batch size.')
    group.add_argument('--train-epochs', type=int, default=3,
                       help='Number of epochs to train.')
    group.add_argument('--max-train-steps', type=int, default=None, help="Max train steps. Will overide train_epochs")
    group.add_argument('--start-epoch', type=int, default=0, help='Start epoch for training.')
    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-scheduler', type=str, default='linear', choices=['linear', 'constant'])
    group.add_argument('--warmup', type=float, default=0.01,
                       help='Percentage of total iterations to warmup on '
                       '(.01 = 1 percent of all training iters).')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument('--train', action='store_true')
    group.add_argument('--dev', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument("--deepspeed", default=None, type=str, help='Config file for deepspeed.')
    group.add_argument('--bf16', action='store_true', help="Training reader use bf16")
    group.add_argument('--gradient-checkpointing', action='store_true', help='Use gradient checkpointing to save memory.')
    group.add_argument('--train-iters-epoch', type=int, default=None, help='For debug.')
    return parser

def _add_dataset_args(parser):
    group = parser.add_argument_group(title='dataset')
    group.add_argument('--qa-file-dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')
    group.add_argument('--qa-file-test', type=str, default=None,
                       help='Path to the QA dataset test file.')
    group.add_argument('--qa-file-train', type=str, default=None,
                       help='Path to the QA dataset train file.')
    group.add_argument('--iterable-load', action='store_true', help='Whether to load json dataset in iterabel way to save memory cost.')
    return parser

def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-step', type=int, default=1000,
                       help='Save checkpoint every intervel.')
    group.add_argument('--save-strategy', type=str, default='epoch', choices=['epoch', 'step', 'no'])
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    return parser

def _add_distributed_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='Size of the model parallel.')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')
    group.add_argument('--eval-strategy', type=str, default='epoch', choices=['epoch', 'step', 'no'], help='Whether evaluate on dev set when epoch ends.')
    group.add_argument('--eval-step', default=1000, type=int, help="Do evaluation every eval_step")
    group.add_argument('--eval-only-loss', action='store_true',
                       help='Only get evaluation loss without any other metric.')
    group.add_argument("--eval-metric", type=str, default="em", choices=["em", "rougel", "recall", "str-em"], help="Evaluation metric")
    group.add_argument('--eval-iters', type=int, default=None, help="Debug")
    return parser

def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--wandb-project', type=str, default=None, help='Wandb project name for logging.')
    group.add_argument('--wandb-name', type=str, default=None, help='Wandb display name for this run.')
    return parser

def _add_reader_args(parser):
    group = parser.add_argument_group(title='unitedqa')
    # checkpointing
    group.add_argument('--t5-model-path', type=str, default=None,
                       help='Directory containing a pre-trained T5 checkpoint (needed to start REALM)')
    group.add_argument('--with-title', action='store_true', help='Whether use title in reader.')
    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help="Maximum encoder sequence length to process.")
    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--reader-dropout', default=0.1, type=float, help='Dropout in reader training.')
    # training
    return parser

def _add_retriever_args(parser):
    group = parser.add_argument_group(title='retriever')
    # retriever
    group.add_argument('--retriever-query-seq-length', type=int, default=128,
                       help="Maximum sequence length to process in question encoder.")
    group.add_argument('--retriever-seq-length', type=int, default=256,
                       help='Maximum sequence length for the context model')
    group.add_argument('--query-encoder-path', type=str, default=None,
                        help='Directory containing a pre-trained DPR query encoder.')
    group.add_argument('--context-encoder-path', type=str, default=None,
                        help='Directory containing a pre-trained DPR context encoder.')
    # faiss index
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--embedding-size', type=int, default=768,
                       help="The hidden size of context vectors.")
    group.add_argument('--embedding-path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding data to/from')
    group.add_argument('--match', type=str, default='string', choices=['regex', 'string', 'none'],
                        help="Answer matching logic type")
    group.add_argument('--topk-retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')
    group.add_argument('--save-topk-outputs-path', type=str, default=None,
                       help='Path of directory to save the topk outputs from retriever')
    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer report progress')
    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence from DPR paper')
    group.add_argument('--log-interval-input-data', type=int, default=100000,
                       help='Report progress while reading wikipeida file.')
    group.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")
    group.add_argument('--evidence-number', type=int, default=None,
                       help='Number of document to encode and search (for DEBUG).')
    return parser


