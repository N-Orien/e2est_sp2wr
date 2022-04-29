#!/bin/bash

# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -m b
#$ -m a
#$ -m e
#$ -cwd

set -eu

export PS1='' # bugfix: https://github.com/conda/conda/issues/8186#issuecomment-532129334
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate lasertagger

### Required parameters (modify before calling the script!) ###

# Preprocessed data and models will be stored here.
OUTPUT_DIR=./output
# Download the pretrained BERT model:
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
BERT_BASE_DIR=/larch/share/bert/Japanese_models/Wikipedia/L-24_H-1024_A-16_E-30_BPE_WWM
# Directory containing input files for prediction
INPUT_DIR=./temp

### Optional parameters ###

# If you train multiple models on the same data, change this label.
EXPERIMENT=experiment_lmt_2019_1st_human_v2_large_ff_zenkaku

###########################


### 4. Prediction

# Get the most recently exported model directory.
TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}

for setname in train dev test; do
  python ../predict_main.py \
  --input_file=${DATA_PATH}/${setname}.tsv \
  --input_format=wikisplit \
  --output_file=${DATA_PATH}/${setname}_pred.tsv \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --saved_model=${SAVED_MODEL_DIR} \
  --enable_swap_tag=false
done
