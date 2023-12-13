#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=50G
#SBATCH --time=0
#SBATCH --partition=babel-shared-long
#SBATCH --time=2-00:00:00

python mecl_experiments/sentiment-analysis/run_train.py  --margin 0.3 \
	--num-instances 1 --batch_size 32 --warmup-step 300 --lr 0.00005 --max_iter 2000 --save-freq 100 \
	--alpha 0.1 --alpha-scheduler step --alpha-milestones 800 1600 \
	--domains books dvd electronics kitchen_\&_housewares \
    	--run_name "distilbert-MECL" \
    --dataset_loc data/sentiment-dataset \
    --train_pct 0.9 \
 --model_dir wandb_local/emnlp_sentiment_experiments/distilbert-MECL \
 --domain-index 0
