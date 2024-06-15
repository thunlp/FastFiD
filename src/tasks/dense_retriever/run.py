from tasks.dense_retriever.validation import OpenRetrievalEvaluator


def main(args):
    evaluator = OpenRetrievalEvaluator(args)
    if args.dev:
        if args.rank == 0:
            print("Evaluate retriever on dev set...")
        evaluator.evaluate(args, "openqa", args.qa_file_dev, 'dev')
    if args.test:
        if args.rank == 0:
            print("Evaluate retriever on test set...")
        evaluator.evaluate(args, "openqa", args.qa_file_test, 'test')
    if args.train:
        if args.rank == 0:
            print("Evaluate retriever on train set...")
        evaluator.evaluate(args, "openqa", args.qa_file_train, 'train')
