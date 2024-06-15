# !/bin/bash

data_dir="datas/qaps_fidkd"

python src/tools/convert_qap_data.py --qa-file-train ${data_dir}/nq-train.json  \
                                     --qa-file-dev ${data_dir}/nq-dev.json \
                                     --qa-file-test ${data_dir}/nq-test.json \
                                     --output-dir datas/qaps_fidkd_text \
                                     --title-answer