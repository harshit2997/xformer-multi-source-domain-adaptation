import argparse
import gc
import os
import random
from typing import AnyStr
from typing import List
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW
from transformers import DistilBertTokenizer
from transformers import AutoModel
# from transformers import get_linear_schedule_with_warmup
import sys
sys.path.append('.')
from datareader import MultiDomainSentimentDataset
from datareader import collate_batch_transformer
from metrics import MultiDatasetClassificationEvaluator, MECLClassificationEvaluator
from metrics import ClassificationEvaluator
from metrics import acc_f1

from metrics import plot_label_distribution
from model import DistilBertFeatureExtractor
from model import *
from sklearn.model_selection import ParameterSampler
from multi_source_trainer_nlp import MultiSourceTrainer

from utils.lr_scheduler import WarmupCosineLR

def train(
        model: torch.nn.Module,
        train_dls: List[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        validation_evaluators: MultiDatasetClassificationEvaluator,
        n_epochs: int,
        device: AnyStr,
        log_interval: int = 1,
        patience: int = 10,
        model_dir: str = "wandb_local",
        gradient_accumulation: int = 1,
        domain_name: str = ''
):
    #best_loss = float('inf')
    best_accs = [0.0]*(len(train_dls)+1)
    patience_counter = 0

    epoch_counter = 0
    total = sum(len(dl) for dl in train_dls)

    # Main loop
    while epoch_counter < n_epochs:
        dl_iters = [iter(dl) for dl in train_dls]
        dl_idx = list(range(len(dl_iters)))
        finished = [0] * len(dl_iters)
        i = 0
        with tqdm(total=total, desc="Training") as pbar:
            while sum(finished) < len(dl_iters):
                random.shuffle(dl_idx)
                for d in dl_idx:
                    domain_dl = dl_iters[d]
                    batches = []
                    try:
                        for j in range(gradient_accumulation):
                            batches.append(next(domain_dl))
                    except StopIteration:
                        finished[d] = 1
                        if len(batches) == 0:
                            continue
                    optimizer.zero_grad()
                    for batch in batches:
                        model.train()
                        batch = tuple(t.to(device) for t in batch)
                        input_ids = batch[0]
                        masks = batch[1]
                        labels = batch[2]
                        # Testing with random domains to see if any effect
                        #domains = torch.tensor(np.random.randint(0, 16, batch[3].shape)).to(device)
                        domains = batch[3]

                        loss, logits = model(input_ids, attention_mask=masks, domains=domains, labels=labels)
                        loss = loss.mean() / gradient_accumulation
                        if i % log_interval == 0:
                            # wandb.log({
                            #     "Loss": loss.item(),
                            #     "alpha0": alpha[:,0].cpu(),
                            #     "alpha1": alpha[:, 1].cpu(),
                            #     "alpha2": alpha[:, 2].cpu(),
                            #     "alpha_shared": alpha[:, 3].cpu()
                            # })
                            wandb.log({
                                "Loss": loss.item()
                            })

                        loss.backward()
                        i += 1
                        pbar.update(1)

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

        gc.collect()

        # Inline evaluation
        for v, validation_evaluator in enumerate(validation_evaluators):
            (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
            print(f"Validation acc {v}: {acc}")

            #torch.save(model.state_dict(), f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}.pth')

            # Saving the best model and early stopping
            #if val_loss < best_loss:
            if acc > best_accs[v]:
                best_accs[v] = acc
                #wandb.run.summary['best_validation_loss'] = best_loss
                if v < len(train_dls):
                    torch.save(model.module.domain_experts[v].state_dict(),
                               f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}_{v}.pth')
                else:
                    torch.save(model.module.shared_bert.state_dict(),
                               f'{model_dir}/{Path(wandb.run.dir).name}/model_{domain_name}_{v}.pth')
                patience_counter = 0
                # Log to wandb
                wandb.log({
                    'Validation accuracy': acc,
                    'Validation Precision': P,
                    'Validation Recall': R,
                    'Validation F1': F1,
                    'Validation loss': val_loss})
            else:
                patience_counter += 1
                # Stop training once we have lost patience
                if patience_counter == patience:
                    break

        gc.collect()
        epoch_counter += 1


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.8)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=1)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    # parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--pretrained_bert", help="Directory with weights to initialize the shared model with", type=str, default=None)
    parser.add_argument("--pretrained_multi_xformer", help="Directory with weights to initialize the domain specific models", type=str, default=None)
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training', default=[])
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--run_name", type=str, help="A name for the run", default="pheme-baseline")
    parser.add_argument("--model_dir", help="Where to store the saved model", default="wandb_local", type=str)
    parser.add_argument("--tags", nargs='+', help='A list of tags for this run', default=[])
    parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="l2 reg", type=float, default=0.01)
    parser.add_argument("--n_heads", help="Number of transformer heads", default=6, type=int)
    parser.add_argument("--n_layers", help="Number of transformer layers", default=6, type=int)
    parser.add_argument("--d_model", help="Transformer model size", default=768, type=int)
    parser.add_argument("--ff_dim", help="Intermediate feedforward size", default=2048, type=int)
    parser.add_argument("--gradient_accumulation", help="Number of gradient accumulation steps", default=1, type=int)
    parser.add_argument("--model", help="Name of the model to run", default="VanillaBert")
    parser.add_argument("--indices_dir", help="If standard splits are being used", type=str, default=None)
    parser.add_argument("--ensemble_basic", help="Use averaging for the ensembling method", action="store_true")
    parser.add_argument('--optimizer', type=str, default='AdamW')
    # parser.add_argument('--scheduler', type=str, default='step_lr', choices=['step_lr', 'cosine_lr'])
    parser.add_argument('--num-instances', type=int, default=1)
    parser.add_argument('--milestones', nargs='+', type=int, default=[4000, 8000])
    parser.add_argument('--domain-index', type=int, default=-1)

    parser.add_argument('--warmup-steps', type=int, default=1000)

    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=2000)
    parser.add_argument('--refresh-freq', type=int, default=1000)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    
    # alpha scheduler
    parser.add_argument('--alpha-scheduler', type=str, default='constant', choices=['step', 'constant'])
    parser.add_argument('--alpha-milestones', nargs='+', type=int, default=[4000, 8000])
    parser.add_argument('--alpha', type=float, default=0.1)

    # uniform loss
    parser.add_argument('--uniform_weight', type=float, default=0.1)
    parser.add_argument('--unif_t', type=float, default=2.0)
    parser.add_argument('--q_size', type=int, default=16)

    parser.add_argument('--re_weight', type=float, default=0.25)

    # parser.add_argument('--validate', action='store_true', help='validation when training')


    args = parser.parse_args()

    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # See if CUDA available
    device = torch.device("cpu")
    if args.n_gpu > 0 and torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # model configuration
    bert_model = 'distilbert-base-uncased'
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    # n_epochs = args.n_epochs


    # wandb initialization
    wandb.init(
        project="multisource-sentiment-emnlp",
        name=args.run_name,
        config={
            # "epochs": n_epochs,
            "learning_rate": lr,
            "warmup": args.warmup_steps,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "train_split_percentage": args.train_pct,
            "bert_model": bert_model,
            "seed": seed,
            "tags": ",".join(args.tags)
        }
    )
    #wandb.watch(model)
    #Create save directory for model
    if not os.path.exists(f"{args.model_dir}/{Path(wandb.run.dir).name}"):
        os.makedirs(f"{args.model_dir}/{Path(wandb.run.dir).name}")

    # Create the dataset
    all_dsets = [MultiDomainSentimentDataset(
        args.dataset_loc,
        [domain],
        DistilBertTokenizer.from_pretrained(bert_model)
    ) for domain in args.domains]

    train_sizes = [int(len(dset) * args.train_pct) for j, dset in enumerate(all_dsets)]
    val_sizes = [len(all_dsets[j]) - train_sizes[j] for j in range(len(train_sizes))]

    accs = []
    Ps = []
    Rs = []
    F1s = []
    # Store labels and logits for individual splits for micro F1
    labels_all = []
    logits_all = []

    accs_avg = []
    Ps_avg = []
    Rs_avg = []
    F1s_avg = []
    # Store labels and logits for individual splits for micro F1
    labels_all_avg = []
    logits_all_avg = []

    run_all_domains = False

    if args.domain_index <0 or args.domain_index >= len(all_dsets):
        print ("Running for all domains")
        run_all_domains = True

    for i in range(len(all_dsets)):  # permutation loop
        domain = args.domains[i]
        test_dset = all_dsets[i]

        if (not run_all_domains) and args.domain_index!=i:
            continue

        print ("Will test on "+str(domain))

        # Override the domain IDs
        k = 0
        for j in range(len(all_dsets)):
            if j != i:
                all_dsets[j].set_domain_id(k)
                k += 1
        test_dset.set_domain_id(k)
        # For test
        #all_dsets = [all_dsets[0], all_dsets[2]]

        # Split the data
        if args.indices_dir is None:
            subsets = [random_split(all_dsets[j], [train_sizes[j], val_sizes[j]])
                       for j in range(len(all_dsets)) if j != i]
        else:
            # load the indices
            dset_choices = [all_dsets[j] for j in range(len(all_dsets)) if j != i]
            subset_indices = defaultdict(lambda: [[], []])
            with open(f'{args.indices_dir}/train_idx_{domain}.txt') as f, \
                    open(f'{args.indices_dir}/val_idx_{domain}.txt') as g:
                for l in f:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][0].append(int(vals[1]))
                for l in g:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][1].append(int(vals[1]))
            subsets = [[Subset(dset_choices[d], subset_indices[d][0]), Subset(dset_choices[d], subset_indices[d][1])] for d in
                       subset_indices]

        samplers = [RandomSampler(subset[0], replacement=True, num_samples=args.max_iter*batch_size) for subset in subsets]

        train_dls = [DataLoader(
            subsets[i][0],
            batch_size=batch_size,
            sampler = samplers[i],
            collate_fn=collate_batch_transformer
        ) for i in range(len(subsets))]

        val_ds = [subset[1] for subset in subsets]

        validation_evaluator = MECLClassificationEvaluator(val_ds, device)
        test_evalator = MECLClassificationEvaluator([test_dset], device)
        ind_val_evaluators = [MECLClassificationEvaluator([v], device) for v in val_ds]

        ##### Create models, schedulers, optimizers
        models = []
        for j in range(len(train_dls)):
            models.append(DistilBertFeatureExtractor(bert_model).cuda())

        model_optimizers = []
        for j in range(len(train_dls)):
            model_optimizers.append(AdamW(models[j].parameters(), lr=lr, weight_decay=weight_decay))

        model_schedulers = []
        for j in range(len(train_dls)):
            model_schedulers.append(WarmupCosineLR(model_optimizers[j], max_iters=args.max_iter, warmup_factor=0.01,
                                          warmup_iters=args.warmup_steps))
            # model_schedulers.append(get_linear_schedule_with_warmup(model_optimizers[j], args.warmup_steps, args.max_iter))

        mlps = []
        for j in range(len(train_dls)):
            mlps.append(MLP(768, 768).cuda())

        mlps_optimizers = []
        for j in range(len(train_dls)):
            mlps_optimizers.append(AdamW(mlps[j].parameters(), lr=lr, weight_decay=weight_decay))

        mlps_schedulers = []
        for j in range(len(train_dls)):
            mlps_schedulers.append(WarmupCosineLR(mlps_optimizers[j], max_iters=args.max_iter, warmup_factor=0.01,
                                          warmup_iters=args.warmup_steps))
            # mlps_schedulers.append(get_linear_schedule_with_warmup(model_optimizers[j], args.warmup_steps, n_epochs))
        
        classifiers = []
        for j in range(len(train_dls)):
            classifiers.append(Classifier(768, 2).cuda())
        
        classifiers_optimizers = []
        for j in range(len(train_dls)):
            classifiers_optimizers.append(AdamW(classifiers[j].parameters(), lr=lr, weight_decay=weight_decay))
        
        classifiers_schedulers = []
        for j in range(len(train_dls)):
            classifiers_schedulers.append(WarmupCosineLR(classifiers_optimizers[j], max_iters=args.max_iter, warmup_factor=0.01,
                                          warmup_iters=args.warmup_steps))
            # classifiers_schedulers.append(get_linear_schedule_with_warmup(model_optimizers[j], args.warmup_steps, n_epochs))

        ##### Call train and return expert model
        trainer = MultiSourceTrainer(models, classifiers, mlps, model_optimizers, classifiers_optimizers, mlps_optimizers, model_schedulers, classifiers_schedulers,mlps_schedulers, args)

        trainer.train_multi_source(zip(*train_dls),validation_evaluator, domain, ind_val_evaluators)
        expert_model = trainer.ema_model
        expert_classifier = trainer.ema_cls

        checkpoint_path = os.path.join(args.model_dir, 'checkpoints', 'best_ema_checkpoint_'+domain+'.pth.tar')
        checkpoint = torch.load(checkpoint_path)
        
        expert_model.eval()
        expert_classifier.eval()
        
        expert_model.load_state_dict(checkpoint['model_state_dict']) 
        expert_classifier.load_state_dict(checkpoint['classifier_state_dict'])

        (loss, acc, P, R, F1), _, (labels, logits) = test_evalator.evaluate(expert_model, expert_classifier, return_labels_logits=True)                    

        print ("Domain "+str(domain)+":")

        print(f"{domain} F1: {F1}")
        print(f"{domain} Accuracy: {acc}")
        print()

        wandb.run.summary[f"{domain}-P"] = P
        wandb.run.summary[f"{domain}-R"] = R
        wandb.run.summary[f"{domain}-F1"] = F1
        wandb.run.summary[f"{domain}-Acc"] = acc
        Ps.append(P)
        Rs.append(R)
        F1s.append(F1)
        accs.append(acc)
        labels_all.extend(labels)
        logits_all.extend(logits)

        # with open(f'{args.model_dir}/{Path(wandb.run.dir).name}/pred_lab.txt', 'a+') as f:
        #     for p, l in zip(np.argmax(logits, axis=-1), labels):
        #         f.write(f'{domain}\t{p}\t{l}\n')

    acc, P, R, F1 = acc_f1(logits_all, labels_all)
    # Add to wandb
    wandb.run.summary[f'test-loss'] = loss
    wandb.run.summary[f'test-micro-acc'] = acc
    wandb.run.summary[f'test-micro-P'] = P
    wandb.run.summary[f'test-micro-R'] = R
    wandb.run.summary[f'test-micro-F1'] = F1

    wandb.run.summary[f'test-macro-acc'] = sum(accs) / len(accs)
    wandb.run.summary[f'test-macro-P'] = sum(Ps) / len(Ps)
    wandb.run.summary[f'test-macro-R'] = sum(Rs) / len(Rs)
    wandb.run.summary[f'test-macro-F1'] = sum(F1s) / len(F1s)