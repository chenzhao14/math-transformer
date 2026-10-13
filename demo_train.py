# Source Acknowledgment
# This material includes code modified from the open-source project https://github.com/karpathy/llama2.c.
# The code is used under the MIT License, available at https://github.com/karpathy/llama2.c/tree/master?tab=MIT-1-ov-file#.


import math


import os
import time
from contextlib import nullcontext
from functools import partial
import torch
from model import Transformer, ModelArgs
from tokenizer import BatchProcessor


# -----------------------------------------------------------------------------
from demo_training_config import Config
from math_addition_config import create_math_addition_config

config = create_math_addition_config(
    pretok_bin_dir="pretok_data/simple_math_range_10",
    model_out_dir="models/math_addition_test",
    vocab_size=18,
    max_iters=500
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
# -----------------------------------------------------------------------------

tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {batch_size} batch size * {max_seq_len} max seq len")
os.makedirs(model_out_dir, exist_ok=True)
torch.manual_seed(1)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# partial creates a new function with some arguments fixed
iter_batches = partial(
    BatchProcessor.iter_batches,
    batch_size=batch_size,
    device=device,
    num_workers=0,
    max_seq_len=max_seq_len,
    pretok_bin_dir=pretok_bin_dir,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
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

# init a new model from scratch
print("Initializing a new model from scratch")
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return coeff * learning_rate

# training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config_dict,
                }
                print(f"saving checkpoint to {model_out_dir}")

                # This checkpoint is intended to resume training or for detailed evaluation where state restoration is crucial.
                torch.save(checkpoint, os.path.join(model_out_dir, "ckpt.pt"))
                
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X, Y)
            loss = model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # logging
    if iter_num % log_interval == 0:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e}")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
