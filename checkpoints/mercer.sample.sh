#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -N rnn_gru_hpc
#PBS -j oe
#PBS -M apd283@nyu.edu
#PBS -l mem=4GB
#PBS -l walltime=4:00:00

module load cuda/7.5.18
module load cudnn/7.0v4.0
module load numpy/intel/1.10.1

cd spinn
. .venv-hpc/bin/activate
cd checkpoints

export PYTHONPATH=../python:$PYTHONPATH
export THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32

export MODEL_FLAGS=" \
--use_gru \
--data_type snli \
--embedding_data_path ../glove/glove.840B.300d.txt \
--log_path ../logs \
--training_data_path ../snli_1.0/snli_1.0_train.jsonl \
--experiment_name rnn_gru_hpc \
--eval_data_path ../snli_1.0/snli_1.0_dev.jsonl \
--ckpt_path rnn_gru.ckpt_best   \
--batch_size 32 \
--embedding_keep_rate 0.852564448733 \
--eval_seq_length 25 \
--init_range 0.005 \
--l2_lambda 4.42556134893e-06 \
--learning_rate 0.00464868093302 \
--model_dim 600 \
--model_type RNN \
--num_sentence_pair_combination_layers 2 \
--semantic_classifier_keep_rate 0.883392584372 \
--seq_length 25 \
--nopredict_use_cell \
--nouse_tracking_lstm \
--noconnect_tracking_comp \
--tracking_lstm_hidden_dim 0 \
--word_embedding_dim 300"

echo "THEANO_FLAGS: $THEANO_FLAGS"
echo "python -m spinn.models.fat_classifier $MODEL_FLAGS"

python -m spinn.models.fat_classifier $MODEL_FLAGS

