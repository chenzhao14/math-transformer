import json
import os
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from dataset_generator import (
    generate_addition_dataset,
    save_dataset_for_training,
    split_train_val,
)
from math_addition_config import create_math_addition_config
from model_eval import ModelArgs, Transformer
from simple_math_tokenizer import SimpleMathTokenizer
from tokenizer import BatchProcessor



from functools import partial

tokenizer = SimpleMathTokenizer()

def run_one(nsamples):
    ### Generate dataset ###
    range_val = 30  # keep your current range; adjust if needed
    data_dir = "data"

    print(f"Generating data for range {range_val} with nsamples={nsamples}...")
    data = generate_addition_dataset(range_val, n_samples=nsamples, seed=42)
    train_data, val_data = split_train_val(data, val_ratio=0.1, seed=42)

    # Save datasets to files
    save_dataset_for_training(train_data, f'{data_dir}/range_{range_val}_{nsamples}/train.txt')
    save_dataset_for_training(val_data, f'{data_dir}/range_{range_val}_{nsamples}/val.txt')

    def tokenize_and_save_dataset(input_file, output_dir, tokenizer, split_name):
        os.makedirs(output_dir, exist_ok=True)
        tokens = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    token_ids = tokenizer.encode(line, add_bos=True, add_eos=True)
                    tokens.extend(token_ids)
        tokens_array = np.array(tokens, dtype=np.uint16)
        output_file = os.path.join(output_dir, f"{split_name}.bin")
        tokens_array.tofile(output_file)
        print(f"Saved {len(tokens)} tokens to {output_file}")
        return len(tokens)

    # Tokenize datasets with Simple Math Tokenizer
    print("Tokenizing datasets with Simple Math Tokenizer...")
    print("=" * 50)
    output_dir = f"pretok/range_{range_val}_{nsamples}"
    input_dir = f"data/range_{range_val}_{nsamples}"
    tokenize_and_save_dataset(f'{input_dir}/train.txt', output_dir, tokenizer, 'train')
    tokenize_and_save_dataset(f'{input_dir}/val.txt', output_dir, tokenizer, 'val')

    ### Training setup ###
    import math, time
    config = create_math_addition_config(
        pretok_bin_dir=output_dir,
        model_out_dir=f"models/math_addition_range_{range_val}_{nsamples}",
        vocab_size=getattr(tokenizer, "vocab_size", tokenizer.get_vocab_size()),
        max_iters=2000
    )

    config_dict = config.to_dict()
    pretok_bin_dir = config.pretok_bin_dir
    model_out_dir = config.model_out_dir
    eval_interval = config.eval_interval
    log_interval = config.log_interval
    eval_iters = config.eval_iters
    eval_only = config.eval_only
    always_save_checkpoint = config.always_save_checkpoint
    init_from = config.init_from
    wandb_log = config.wandb_log
    wandb_project = config.wandb_project
    wandb_run_name = config.wandb_run_name
    batch_size = config.batch_size
    max_seq_len = config.max_seq_len
    vocab_size = config.vocab_size
    dim = config.dim
    n_layers = config.n_layers
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    multiple_of = config.multiple_of
    dropout = config.dropout
    gradient_accumulation_steps = config.gradient_accumulation_steps
    learning_rate = config.learning_rate
    max_iters = config.max_iters
    weight_decay = config.weight_decay
    beta1 = config.beta1
    beta2 = config.beta2
    grad_clip = config.grad_clip
    decay_lr = config.decay_lr
    warmup_iters = config.warmup_iters
    device = config.device
    dtype = config.dtype

    tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {batch_size} batch size * {max_seq_len} max seq len")
    os.makedirs(model_out_dir, exist_ok=True)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    iter_batches = partial(
        BatchProcessor.iter_batches,
        batch_size=batch_size,
        device=device,
        num_workers=0,
        max_seq_len=max_seq_len,
        pretok_bin_dir=pretok_bin_dir,
    )

    iter_num = 0
    best_val_loss = 1e9

    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            batch_iter = iter_batches(split=split)
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = next(batch_iter)
                with ctx:
                    logits = model(X, Y)
                    loss = model.last_loss
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return coeff * learning_rate

    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)
    t0 = time.time()
    local_iter_num = 0

    patience = 5            # number of eval cycles without improvement
    min_delta = 0.0         # required improvement to reset patience
    no_improve = 0
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        if iter_num == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict()}, "checkpoints/ckpt_init.pt")

        if iter_num == max_iters // 4:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict()}, "checkpoints/ckpt_mid.pt")

        if iter_num == 500 - 1:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict()}, "checkpoints/ckpt_final.pt")

        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # early stopping bookkeeping
            if losses["val"] < (best_val_loss - min_delta):
                best_val_loss = losses["val"]
                no_improve = 0
            else:
                no_improve += 1

            # save checkpoint
            if losses["val"] <= best_val_loss or always_save_checkpoint:
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config_dict,
                    }
                    print(f"saving checkpoint to {model_out_dir}")
                    torch.save(checkpoint, os.path.join(model_out_dir, 'ckpt.pt'))

            # trigger early stop
            if no_improve >= patience:
                print(f"Early stopping: no val improvement for {patience} evals.")
                break

        if iter_num == 0 and eval_only:
            return

        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits = model(X, Y)
                loss = model.last_loss / gradient_accumulation_steps
            X, Y = next(train_batch_iter)
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e}")
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

if __name__ == "__main__":
    run_one(500)  
    #### Testing ####
    from test import evaluate_on_val,  print_examples_for_range_sample
    print_examples_for_range_sample(30, 500, k=10)
