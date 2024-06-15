import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import logging
from arguments import parse_args
from utils import *
from tasks.dense_retriever.run import main as retriever_main
from tasks.reader.run import main as reader_main
from tasks.end2endqa.run import main as end2endqa_main
try:
    from tasks.bmreader.run import main as bmqa_main
except ModuleNotFoundError:
    bmqa_main = None

logger = logging.getLogger("Tasks")

def _add_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True, choices=['retrieval', 'qa', 'end2endqa', 'bmqa'], help='Task name.')
    # add qa args
    group.add_argument('--output-prediction-path', type=str, default=None,
                       help='Path to save predictions. Default is none and will not save')
    group.add_argument('--time-analysis', action='store_true', help='time analysis for each training step.')
    group.add_argument('--with-extractive-loss', action='store_true', help='Add extractive loss in reader training.')
    group.add_argument('--with-generative-loss', action='store_true', help='Add generative loss in reader training.')
    group.add_argument('--with-rerank-loss', action='store_true', help='Ablation study on rerank passage. SHOULD NOT be used with extraction loss.')
    group.add_argument('--extractive-loss-lambda', type=float, default=0.5, help='coefficient of extraction loss')
    group.add_argument('--extractive-loss-temperature', type=float, default=1.0, help='temperature of extractive loss')
    group.add_argument('--with-pdr-loss', action='store_true', help='Add pdr loss in extractive reader training.')
    group.add_argument('--with-passage-loss', action='store_true', help='Add passage loss in extractive training.')
    group.add_argument('--fid-light-k', type=int, default=None, help="The k vector to use in FiD-Light (https://arxiv.org/pdf/2209.14290.pdf), used when inference-method is fid_light_generative.")
    group.add_argument('--inference-method', default='generative', choices=['generative', 'extractive', 'select_generative', 'rerank_generative', 'fid_light_generative'])
    group.add_argument('--support-sentence-length', type=int, default=None,
                       help="Maximum sequence length for support sentence.")
    group.add_argument('--support-sentence-topk-accuracies', nargs='+', type=int, default=[],
                       help="Which top-k accuracies to report for sentence extraction (e.g. '1 5 20')")
    group.add_argument('--rerank-topk-accuracies', nargs='+', type=int, default=[])
    group.add_argument('--inference-sum', action='store_true', help='Whether to add probability of same string for extractive model')
    group.add_argument('--freeze-extraction', action='store_true', help='Whether to freeze extraction part when use select inference training.')
    group.add_argument('--record-cross-attention', action='store_true', help='Whether to return cross attentions for analyse')
    group.add_argument('--disable-tqdm', action='store_true', help="Disable tqdm in evaluation.")
    group.add_argument('--synced-gpus', action='store_true', help="Whether to continue running the while loop until max_length. Need to be true under DeepSpeed ZeRO3.")
    group.add_argument('--num-beams', type=int, default=1, help="Beam size for generative inference.")
    group.add_argument('--format-type', type=int, default=0, help="format for llama model, 0 for question in the end, 1 for question in the beginning.")
    return parser


if __name__ == "__main__":
    args = parse_args(extra_args_provider=_add_tasks_args)
    logger_config(args)
    if args.task == 'retrieval':
        distributed_init(args)
        retriever_main(args)
    elif args.task == 'qa':
        distributed_init(args)
        reader_main(args)
    elif args.task == 'end2endqa':
        distributed_init(args)
        end2endqa_main(args)
    elif args.task == 'bmqa':
        if bmqa_main is None:
            raise ValueError("Please install BMTrain and ModelCenter")
        bmqa_main(args)
    else:
        raise ValueError()

