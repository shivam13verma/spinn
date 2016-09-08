#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -N spinn_valid_predict_mercer
#PBS -j oe
#PBS -M apd283@nyu.edu
#PBS -l mem=6GB
#PBS -l walltime=16:00:00

module load cuda/7.5.18
module load cudnn/7.0v4.0
module load numpy/intel/1.10.1

cd spinn
. .venv-hpc/bin/activate
cd checkpoints

export PYTHONPATH=../python:$PYTHONPATH
export THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32

export MODEL_FLAGS=" \
--data_type snli \
--validate_transitions \
--model_type Model1 \
--training_data_path ../snli_1.0/snli_1.0_train.jsonl \
--eval_data_path ../snli_1.0/snli_1.0_dev.jsonl \
--embedding_data_path ../glove/glove.840B.300d.txt \
--log_path ../logs \
--experiment_name spinn_valid_predict_mercer \
--ckpt_path spinn_valid_predict_mercer.ckpt_best   \
--word_embedding_dim 300 \
--batch_size 32 \
--model_dim 600 \
--tracking_lstm_hidden_dim 44 \
--eval_seq_length 50 \
--seq_length 50 \
--init_range 0.005 \
--l2_lambda 2.76018187539e-05 \
--learning_rate 0.00103428201391 \
--semantic_classifier_keep_rate 0.949455648614 \
--transition_cost_scale 0.605159568546 \
"

echo "THEANO_FLAGS: $THEANO_FLAGS"
echo "python -m spinn.models.fat_classifier $MODEL_FLAGS"

python -m spinn.models.fat_classifier $MODEL_FLAGS

