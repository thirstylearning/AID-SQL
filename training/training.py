import time
import torch
import random
import torch.nn as nn
import numpy as np
import os

from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

from training.my_batcher import MyBatcher

from infra import ColBERTConfig

from utils.amp import MixedPrecisionManager
from parameters import DEVICE

from modeling.colbert import ColBERT # my_colbert

from utils.utils import print_message
from training.utils import print_progress, manage_checkpoints

from infra import Run


def train(config: ColBERTConfig, triples, rag_collection_path=None, rag_dataset_path=None, n_epochs: int = 1):
    config.checkpoint = config.checkpoint or "bert-base-uncased"

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if rag_dataset_path is not None:
        if config.reranker:
            raise NotImplementedError()
        else:
            # reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
            reader = MyBatcher(
                config,
                triples,
                rag_collection_path,
                rag_dataset_path,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        raise NotImplementedError()

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert, device_ids=[config.rank], output_device=config.rank, find_unused_parameters=True
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.warmup, num_training_steps=config.maxsteps
        )

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0
    global_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    run_base_dir = Run().path_
    tensorboard_dir = os.path.join(run_base_dir, "tensorboard")
    tb_writer = SummaryWriter(log_dir=tensorboard_dir) if config.rank < 1 else None
    
    last_ckpt_path = None
    start_epoch = config.start_epoch

    for epoch in range(start_epoch, start_epoch + n_epochs):
        start_batch_idx = 0
        reader.shuffle()
        
        eval_top1_scores_in_target = 0
        eval_count = 0

        for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
            if (warmup_bert is not None) and warmup_bert <= batch_idx:
                set_bert_grad(colbert, True)
                warmup_bert = None

            this_batch_loss = 0.0
            # this_batch_ib_loss = 0.0
            # this_batch_self_loss = 0.0
            global_batch_idx += 1

            for batch in BatchSteps:
                with amp.context():
                    try:
                        queries, passages, target_scores = batch
                        encoding = [queries, passages]
                    except:
                        encoding, target_scores = batch
                        encoding = [encoding.to(DEVICE)]

                    scores = colbert(*encoding)

                    if config.use_ib_negatives or config.use_self_negatives or config.use_self_easy_negatives:
                        scores, ib_self_loss = scores
                        ib_loss, self_loss, self_easy_loss = ib_self_loss
                    
                    # use first query's scores for training evaluation
                    train_eval_scores = scores[: config.nway]
                    trian_eval_target_scores = torch.tensor(target_scores[: config.nway]).to(DEVICE)
                    # count how many top 1 scores are in the top 1 target scores
                    top_1_scores = torch.topk(train_eval_scores, 1, dim=-1).indices
                    top_1_target_scores = torch.topk(trian_eval_target_scores, 1, dim=-1).indices
                    if top_1_scores[0] in top_1_target_scores:
                        eval_top1_scores_in_target += 1
                    eval_count += 1

                    scores = scores.view(-1, config.nway)
                    
                    loss = torch.tensor(0.0).to(DEVICE)

                    if len(target_scores) and not config.ignore_scores and config.use_scores_distillation:
                        target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                        target_scores = target_scores * config.distillation_alpha
                        target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                        loss += torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(log_scores, target_scores)
                        
                    # if config.rank < 1:
                    #     scores_distillation_loss = loss.item()

                    if config.use_ib_negatives:
                        if config.rank < 1:
                            print("\t\t\t\t", loss.item(), ib_loss.item())

                        loss += ib_loss
                        
                    if config.use_self_negatives:
                        if config.rank < 1:
                            print("\t\t\t\t", loss.item(), self_loss.item())
                        
                        loss += self_loss
                        
                    if config.use_self_easy_negatives:
                        if config.rank < 1:
                            print("\t\t\t\t", loss.item(), self_easy_loss.item())
                        
                        loss += self_easy_loss

                    loss = loss / config.accumsteps

                if config.rank < 1:
                    print_progress(scores)

                amp.backward(loss)

                this_batch_loss += loss.item()

            train_loss = this_batch_loss if train_loss is None else train_loss
            train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

            amp.step(colbert, optimizer, scheduler)

            if config.rank < 1:
                print_message(batch_idx, train_loss)
                # manage_checkpoints(config, colbert, optimizer, batch_idx + 1, savepath=None)
                # this_batch_ib_loss = ib_loss.item() if config.use_ib_negatives else 0.0
                # this_batch_self_loss = self_loss.item() if config.use_self_negatives else 0.0

                tb_writer.add_scalar("Loss/train", train_loss, global_batch_idx - 1)
                # tb_writer.add_scalar("Loss/scores_distillation_loss", scores_distillation_loss, global_batch_idx - 1)
                # tb_writer.add_scalar("Loss/ib_loss", this_batch_ib_loss, global_batch_idx - 1)
                # tb_writer.add_scalar("Loss/self_loss", this_batch_self_loss, global_batch_idx - 1)

        if config.rank < 1:
            print_message(f"#> Done with epoch {epoch}")
            
            # write eval to tensorboard
            tb_writer.add_scalar("eval/training", eval_top1_scores_in_target / eval_count, epoch + 1)

            if (epoch + 1) % config.save_each_n_epoch != 0:
                continue
            
            this_epoch_ckpt_path = os.path.join(run_base_dir, f"epoch_{epoch + 1}", "checkpoints")
            manage_checkpoints(
                config, colbert, optimizer, batch_idx + 1, savepath=this_epoch_ckpt_path, consumed_all_triples=True
            )
            last_ckpt_path = this_epoch_ckpt_path

    if config.rank < 1:
        print_message("#> Done with all triples!")
        # ckpt_path = manage_checkpoints(
        #     config, colbert, optimizer, batch_idx + 1, savepath=None, consumed_all_triples=True
        # )

        tb_writer.close()
        
        return last_ckpt_path

        # return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
