from tasks.reader.trainer import Trainer
import logging
import os
import pandas as pd
import json
# from data.data_utils import count_sentence_length

logger = logging.getLogger("reader-main")

def write_generation(args, trainer, results, dev_metrics, test_metrics):
    predictions = results[1]
    os.makedirs(args.output_prediction_path, exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.qa_file_test))[0] + '.csv'
    to_write_datas = {'question': [], 'answers': [], 'prediction': []}
    for index, instance in enumerate(trainer.datasets['test'].samples):
        if index >= len(predictions):
            break
        to_write_datas['question'].append(instance['question'])
        to_write_datas['answers'].append(instance['answers'])
        for key in predictions[index]:
            to_write_datas[key].append(predictions[index][key].replace("\n", "\\n"))
    pd_data = pd.DataFrame.from_dict(to_write_datas)
    output_path = os.path.join(args.output_prediction_path, filename)
    pd_data.to_csv(output_path, index=False)
    logger.info("Write prediction to {}".format(output_path))
    with open(os.path.join(args.output_prediction_path, 'metrics.txt'), 'w') as f:
        if args.dev:
            f.write("Evaluate Results | " + str(dev_metrics) + '\n')
        f.write("Test Results | " + str(test_metrics) + '\n') 

def write_support_sentence(args, trainer, results, dev_metrics, test_metrics):
    predictions = results[1]
    os.makedirs(args.output_prediction_path, exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.qa_file_test))[0] + '-extraction' + '.json'
    to_write_datas = []
    for index, instance in enumerate(trainer.datasets['test'].samples):
        if index >= len(predictions):
            break
        write_data = {'question': instance['question'], 'answers': instance['answers']}
        for rank, p in enumerate(predictions[index]):
            p['rank'] = rank + 1
        write_data['predict_support_sentences'] = predictions[index]
        to_write_datas.append(write_data)
    output_path = os.path.join(args.output_prediction_path, filename)
    json.dump(to_write_datas, open(output_path, 'w'), indent=4, ensure_ascii=False)
    logger.info("Write prediction to {}".format(output_path))
    with open(os.path.join(args.output_prediction_path, 'metrics.txt'), 'w') as f:
        if args.dev:
            f.write("Evaluate Results | " + str(dev_metrics) + '\n')
        f.write("Test Results | " + str(test_metrics) + '\n') 

def log_metrics(metrics, prefix=""):
    logger.info(prefix)
    key_len = max(len(key) for key in metrics.keys())
    for key, v in metrics.items():
        key = key + " " * (key_len - len(key))
        if isinstance(v, float):
            logger.info(f"{key}: {v:.4f}")
        else:
            logger.info(f"{key}: {v}")

def main(args):
    trainer = Trainer(args)
    # print("dev:")
    # count_sentence_length(trainer.datasets['dev'])
    # print("test:")
    # count_sentence_length(trainer.datasets['test'])
    # quit()
    if args.train:
        trainer.train(args)
    if args.dev:
        if args.inference_method == 'extractive':
            results = trainer.evaluate_extractive(args, dataset=trainer.datasets['dev'], return_predictions=False)
        else:
            results = trainer.evaluate(args, dataset=trainer.datasets['dev'], return_predictions=False)
        metrics = results[0]
        log_metrics(metrics, "Dev Results")
        dev_metrics = metrics
    if args.test:
        return_predictions = (args.output_prediction_path is not None)
        if args.inference_method == 'extractive':
            results = trainer.evaluate_extractive(args, dataset=trainer.datasets['test'], return_predictions=return_predictions)
            metrics = results[0]
            log_metrics(metrics, "Test Results")
            if return_predictions and args.rank == 0:
                write_support_sentence(args, trainer, results, dev_metrics=dev_metrics if args.dev else None, test_metrics=metrics)
        else:
            results = trainer.evaluate(args, dataset=trainer.datasets['test'], return_predictions=return_predictions)
            metrics = results[0]
            log_metrics(metrics, "Test Results")
            if return_predictions and args.rank == 0:
                write_generation(args, trainer, results, dev_metrics=dev_metrics if args.dev else None, test_metrics=metrics)
        # if args.inference_method == 'select_generative':
        #     results = trainer.evaluate(args, dataset=trainer.datasets['test'], return_predictions=return_predictions)
        #     metrics = results[0]
        #     logger.info("Test Results | " + str(metrics))
        #     if return_predictions and args.rank == 0:
        #         write_generation(args, trainer, results, dev_metrics=dev_metrics if args.dev else None, test_metrics=metrics) 
        # if args.inference_method == 'rerank_generative':
        #     results = trainer.evaluate(args, dataset=trainer.datasets['test'], return_predictions=return_predictions)
        #     metrics = results[0]
        #     logger.info("Test Results | " + str(metrics))
        #     if return_predictions and args.rank == 0:
        #         write_generation(args, trainer, results, dev_metrics=dev_metrics if args.dev else None, test_metrics=metrics) 
        
