import json
import random

filepath = "./datas/qaps_fidkd_text/nq-test.json"
outputpath_format = "./datas/qaps_fidkd_text/nq-test-s{seed}-{sample_number}.json"
sample_number = 200
seed = 2023
outputpath = outputpath_format.format(seed=seed, sample_number=sample_number)
if __name__ == "__main__":
    with open(filepath) as f:
        data = json.load(f)
    idxs = list(range(len(data)))
    random.Random(seed).shuffle(idxs)
    select_idxs = idxs[:sample_number]
    select_datas = []
    for index in select_idxs:
        select_datas.append(data[index])
    with open(outputpath, 'w') as f:
        json.dump(select_datas, f, indent=4, ensure_ascii=False)
