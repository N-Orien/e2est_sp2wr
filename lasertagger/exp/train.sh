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

# You can download the WikiSplit data from:
# https://github.com/google-research-datasets/wiki-split
DATA_DIR=../data_conv
# Preprocessed data and models will be stored here.
OUTPUT_DIR=./output
# Download the pretrained BERT model:
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
BERT_BASE_DIR=/larch/share/bert/Japanese_models/Wikipedia/L-24_H-1024_A-16_E-30_BPE_WWM

### Optional parameters ###

# If you train multiple models on the same data, change this label.
EXPERIMENT=experiment_lmt_2019_1st_human_v2_large_ff_zenkaku
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
NUM_EPOCHS=20.0
BATCH_SIZE=8
PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
SAVE_CHECKPOINT_STEPS=1384 # 11072 / 8 = 1384 ... 0

###########################


### 1. Phrase Vocabulary Optimization

python ../phrase_vocabulary_optimization.py \
  --input_file=${WIKISPLIT_DIR}/train.tsv \
  --input_format=wikisplit \
  --vocabulary_size=${PHRASE_VOCAB_SIZE} \
  --max_input_examples=${MAX_INPUT_EXAMPLES} \
  --output_file=${OUTPUT_DIR}/label_map.txt \
  --enable_swap_tag=false


### 2. Converting Target Texts to Tags

python ../preprocess_main.py \
  --input_file=${WIKISPLIT_DIR}/dev.tsv \
  --input_format=wikisplit \
  --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --output_arbitrary_targets_for_infeasible_examples=true \
  --enable_swap_tag=false

python ../preprocess_main.py \
    --input_file=${WIKISPLIT_DIR}/train.tsv \
    --input_format=wikisplit \
    --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
    --output_arbitrary_targets_for_infeasible_examples=false \
    --enable_swap_tag=false


### 3. Model Training

NUM_TRAIN_EXAMPLES=$(cat "${OUTPUT_DIR}/train.tf_record.num_examples.txt")
NUM_EVAL_EXAMPLES=$(cat "${OUTPUT_DIR}/tune.tf_record.num_examples.txt")
CONFIG_FILE=../configs/lasertagger_large_config.json

python ../run_lasertagger.py \
  --training_file=${OUTPUT_DIR}/train.tf_record \
  --eval_file=${OUTPUT_DIR}/tune.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
  --init_checkpoint=${BERT_BASE_DIR}/model.ckpt \
  --do_train=true \
  --do_eval=true \
  --train_batch_size=${BATCH_SIZE} \
  --save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
  --num_train_epochs=${NUM_EPOCHS} \
  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
  --num_eval_examples=${NUM_EVAL_EXAMPLES}


### 4. Prediction

# Export the model.
python ../run_lasertagger.py \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
  --do_export=true \
  --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export

# Get the most recently exported model directory.
TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}
PREDICTION_FILE=${OUTPUT_DIR}/models/${EXPERIMENT}/pred.tsv

python ../predict_main.py \
  --input_file=${WIKISPLIT_DIR}/test.tsv \
  --input_format=wikisplit \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --saved_model=${SAVED_MODEL_DIR} \
  --enable_swap_tag=false


### 5. Evaluation

python ../score_main.py --prediction_file=${PREDICTION_FILE}
