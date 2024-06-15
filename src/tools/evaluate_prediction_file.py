import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from tasks.reader.eval_utils import exact_match_score, metric_max_over_ground_truths
import pandas as pd
import re


if __name__ == "__main__":
    # prediction_file = "./checkpoints/generative_top25_constantlr_title/prediction_top25_generate_epoch9/nq-test.csv"
    prediction_file = "./checkpoints/extractive_top25_mediumlr_title_2/prediction_top25_global_epoch9/nq-test.csv"
    result = pd.read_csv(prediction_file)
    ems = []
    for index in result.index:
        answers = eval(result.loc[index, 'answers'])
        prediction = result.loc[index, 'prediction']
        ems.append(metric_max_over_ground_truths(exact_match_score, prediction, answers))
        if index == 552:
            print(result.loc[index, 'question'])
            break
    em = sum(ems) / len(ems)
    print(em)