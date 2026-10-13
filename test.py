import os
import torch
from model import ModelArgs, Transformer
from simple_math_tokenizer import SimpleMathTokenizer
import random


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt.get("model_args")
    if model_args is None:
        cfg = ckpt.get("config", {})
        model_args = cfg.get("model_args", {})
    model = Transformer(ModelArgs(**model_args)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def parse_val_line(line: str):
    # Expect "a + b = c"
    line = line.strip()
    if not line:
        return None, None
    if "=" in line:
        left, right = line.split("=", 1)
        prompt = left.strip() + " ="
        expected = right.strip()
        return prompt, expected
    return line, ""

@torch.no_grad()
def eval_accuracy_on_file(model, tokenizer, val_path, device="cpu", max_new_tokens=12):
    total, correct = 0, 0
    eos_id = tokenizer.token_to_id('<eos>')

    with open(val_path, "r", encoding="utf-8") as f:
        for raw in f:
            prompt, expected = parse_val_line(raw)
            if not prompt:
                continue
            total += 1

            ids = tokenizer.encode(prompt + " ", add_bos=True, add_eos=False)
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

            out = []
            for _ in range(max_new_tokens):
                logits = model(x)  # forward without targets
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                if eos_id is not None and next_id == eos_id:
                    break
                out.append(next_id)
                x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

            pred_text = tokenizer.decode(out).strip()
            # compare numerically (digits only)
            pred_digits = "".join(ch for ch in pred_text if ch.isdigit())
            exp_digits = "".join(ch for ch in expected if ch.isdigit())
            if pred_digits and exp_digits and pred_digits == exp_digits:
                correct += 1

    acc = correct / total if total else 0.0
    print(f"Accuracy on {os.path.basename(val_path)}: {correct}/{total} = {acc:.2%}")
    return acc

def evaluate_on_val(ckpt_path: str, val_path: str, device: str = "cpu", max_new_tokens: int = 12):
    tokenizer = SimpleMathTokenizer()
    model = load_model(ckpt_path, device=device)
    return eval_accuracy_on_file(model, tokenizer, val_path, device=device, max_new_tokens=max_new_tokens)

# Optional helper for your directory pattern
def evaluate_for_range_sample(range_val: int, nsamples: int,
                              model_dir_fmt="models/math_addition_range_{r}_{n}",
                              data_dir_fmt="data/range_{r}_{n}",
                              device: str = "cpu"):
    ckpt = os.path.join(model_dir_fmt.format(r=range_val, n=nsamples), "ckpt.pt")
    val_path = os.path.join(data_dir_fmt.format(r=range_val, n=nsamples), "val.txt")
    return evaluate_on_val(ckpt, val_path, device=device)




@torch.no_grad()
def predict_text(model, tokenizer, prompt: str, device="cpu", max_new_tokens=12):
    eos_id = tokenizer.token_to_id('<eos>')
    ids = tokenizer.encode(prompt + " ", add_bos=True, add_eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    out = []
    for _ in range(max_new_tokens):
        logits = model(x)  # forward without targets
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        if eos_id is not None and next_id == eos_id:
            break
        out.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    return tokenizer.decode(out).strip()

@torch.no_grad()
def print_examples_on_file(model, tokenizer, val_path, k=10, device="cpu", max_new_tokens=12, seed=0, random_sample=True):
    """Print k examples: question, truth, prediction (+ correctness)."""
    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for raw in f:
            prompt, expected = parse_val_line(raw)
            if prompt:
                samples.append((prompt, expected))
    if not samples:
        print(f"No valid lines found in {val_path}")
        return

    rng = random.Random(seed)
    if random_sample and len(samples) > k:
        picks = rng.sample(samples, k)
    else:
        picks = samples[:k]

    print(f"\nExamples from {os.path.basename(val_path)} (k={len(picks)}):")
    correct_count = 0
    for i, (prompt, expected) in enumerate(picks, 1):
        pred = predict_text(model, tokenizer, prompt, device=device, max_new_tokens=max_new_tokens)
        # numeric-robust compare
        pred_digits = "".join(ch for ch in pred if ch.isdigit())
        exp_digits = "".join(ch for ch in expected if ch.isdigit())
        ok = bool(pred_digits and exp_digits and pred_digits == exp_digits)
        correct_count += int(ok)
        print(f"{i:2d}) Q: {prompt}")
        print(f"    Truth: {expected}")
        print(f"    Pred : {pred} {'✅' if ok else '❌'}")
    print(f"\nShown {len(picks)} examples | correct {correct_count}/{len(picks)}")

def print_examples_for_range_sample(range_val: int, nsamples: int,
                                    model_dir_fmt="models/math_addition_range_{r}_{n}",
                                    data_dir_fmt="data/range_{r}_{n}",
                                    k=10, device="cpu", max_new_tokens=12, seed=0):
    """Helper using your directory pattern."""
    ckpt = os.path.join(model_dir_fmt.format(r=range_val, n=nsamples), "ckpt.pt")
    val_path = os.path.join(data_dir_fmt.format(r=range_val, n=nsamples), "val.txt")
    tokenizer = SimpleMathTokenizer()
    model = load_model(ckpt, device=device)
    print_examples_on_file(model, tokenizer, val_path, k=k, device=device, max_new_tokens=max_new_tokens, seed=seed)




if __name__ == "__main__":
    # Example usage:
    # Evaluate the model trained for range 30 with 50000 samples
    acc = evaluate_for_range_sample(30, 50000)
    # Print some examples from the validation set
    print_examples_for_range_sample(30, 50000, k=10)    
    evaluate_on_val("models/math_addition_range_30_50000/ckpt.pt", "data/range_30_50000/val.txt", device="cpu", max_new_tokens=12)
