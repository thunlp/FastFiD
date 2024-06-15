import argparse
from tqdm import tqdm
from data.llama_qap_dataset import get_qap_dataset
from models.llama.tokenization_llama_fast import LlamaTokenizerFast
from utils import logger_config


args = {
    "model_path": "/public/public/open_source_model/Llama-2-7b-hf",
    "qa_file_train": "../datas/qaps_fidkd_text_sentence/nq-train.json",
    "qa_file_dev": "../datas/qaps_fidkd_text_sentence/nq-dev.json",
    "qa_file_test": "../datas/qaps_fidkd_text_sentence/nq-test.json",
    "train_length_file": "../datas/qaps_fidkd_text_sentence/nq-train-llama-20-len.csv",
    "dev_length_file": "../datas/qaps_fidkd_text_sentence/nq-dev-llama-20-len.csv",
    "test_length_file": "../datas/qaps_fidkd_text_sentence/nq-test-llama-20-len.csv",
    "write_length_file": True,
    "iterable_load": True,
    "topk_retrievals": 20,
    "encoder_seq_length": 4096,
    "decoder_seq_length": 30,
    "with_title": True,
    "rank": 0,
}

if __name__ == "__main__":
    args = argparse.Namespace(**args)
    logger_config(args)
    length_files = {
        'train': args.train_length_file,
        'dev': args.dev_length_file,
        'test': args.test_length_file,
    }
    if args.write_length_file:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.model_path)
        datasets = get_qap_dataset(args, tokenizer, None, None, False)
        for key in datasets:
            length_file = length_files[key]
            print(f"Writing length file for {key} to {length_file} ...")
            with open(length_file, "w") as f:
                f.write("id,length\n")
                index = 0
                for d in tqdm(datasets[key]):
                    total_len = d["total_len"]
                    f.write(f"{index},{total_len}\n")
                    index += 1
    for key in length_files:
        length_file = length_files[key]
        print(f"Reading length file for {key} from {length_file} ...")
        with open(length_file, "r") as f:
            cnt = 0
            num = 0
            for linenum, line in enumerate(f):
                if linenum == 0:
                    continue
                id, length = line.strip().split(",")
                length = int(length)
                num += 1
                if length > args.encoder_seq_length:
                    cnt += 1
            print(f"{key} dataset > {args.encoder_seq_length}: {cnt}/{num} ({cnt / num})")
            
    
    
            
        
    