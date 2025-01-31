#!/bin/bash

. activate xformer-multisource-domain-adaptation

. setenv.sh

run_name="(emnlp-sentiment)"
model_dir="wandb_local/emnlp_sentiment_experiments"
tags="emnlp sentiment experiments"
for i in 1000,1 1001,2 666,3 7,4 50,5; do IFS=","; set -- $i;
  # 1) Basic
  python emnlp_final_experiments/sentiment-analysis/train_basic.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "basic-distilbert-${2}" \
    --model_dir ${model_dir}/basic_distilbert \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003
  indices_dir=`ls -d -t ${model_dir}/basic_distilbert/*/ | head -1`

  # 2) Adv-6
  python emnlp_final_experiments/sentiment-analysis/train_basic_domain_adversarial.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-adversarial-6-${2}" \
    --model_dir ${model_dir}/distilbert_adversarial_6 \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --supervision_layer 6 \
    --indices_dir ${indices_dir}

  # 3) Adv-3
  python emnlp_final_experiments/sentiment-analysis/train_basic_domain_adversarial.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-adversarial-3-${2}" \
    --model_dir ${model_dir}/distilbert_adversarial_3 \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --supervision_layer 3 \
    --indices_dir ${indices_dir}

  # 4, 5) Independent-Avg and Independent-Ft
  python emnlp_final_experiments/sentiment-analysis/train_multi_view_averaging_individuals.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-ensemble-averaging-individuals-${2}" \
    --model_dir ${model_dir}/distilbert_ensemble_averaging_individuals \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --indices_dir ${indices_dir}
  avg_model=`ls -d -t ${model_dir}/distilbert_ensemble_averaging_individuals/*/ | head -1`

  # 6) MoE-DC
  python emnlp_final_experiments/sentiment-analysis/train_multi_view_domainclassifier_individuals.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-ensemble-domainclassifier-individuals-${2}" \
    --model_dir ${model_dir}/distilbert_ensemble_domainclassifier_individuals \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --indices_dir ${indices_dir} \
    --pretrained_model ${avg_model}

  # 7) MoE-Avg
  python emnlp_final_experiments/sentiment-analysis/train_multi_view.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-ensemble-averaging-${2}" \
    --model_dir ${model_dir}/distilbert_ensemble_averaging \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --ensemble_basic \
    --indices_dir ${indices_dir}

  # 8) MoE-Att
  python emnlp_final_experiments/sentiment-analysis/train_multi_view.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-ensemble-attention-${2}" \
    --model_dir ${model_dir}/distilbert_ensemble_attention \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --indices_dir ${indices_dir}

  # 9) MoE-Att-Adv-6
  python emnlp_final_experiments/sentiment-analysis/train_multi_view_domain_adversarial.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-ensemble-attention-adversarial-6-${2}" \
    --model_dir ${model_dir}/distilbert_ensemble_attention_adversarial_6 \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --supervision_layer 6 \
    --indices_dir ${indices_dir}

  # 10) MoE-Att-Adv-3
  python emnlp_final_experiments/sentiment-analysis/train_multi_view_domain_adversarial.py \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
    --n_gpu 1 \
    --n_epochs 5 \
    --domains books dvd electronics kitchen_\&_housewares \
    --seed ${1} \
    --run_name "distilbert-ensemble-attention-adversarial-3-${2}" \
    --model_dir ${model_dir}/distilbert_ensemble_attention_adversarial_4 \
    --tags ${tags} \
    --batch_size 8 \
    --lr 0.00003 \
    --supervision_layer 3 \
    --indices_dir ${indices_dir}

done
