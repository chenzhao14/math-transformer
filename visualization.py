import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from simple_math_tokenizer import SimpleMathTokenizer
from model_eval import Transformer, ModelArgs  


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        # fallback if model_args is not in the checkpoint
        model_args = dict(
            dim=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=4,
            vocab_size=18,
            multiple_of=32,
            max_seq_len=256,
            dropout=0.0,
        )
    model = Transformer(ModelArgs(**model_args))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


def visualize_all_layers(model, tokenizer, sentence, device, head=0, title_prefix=""):
    input_ids = torch.tensor([tokenizer.encode(sentence, add_bos=True, add_eos=True)], device=device)

    with torch.no_grad():
        logits, attn_list = model(input_ids, return_attn=True)

    # Convert ids back to tokens
    tokens = [tokenizer.id_to_token[i] for i in input_ids[0].tolist()]

    n_layers = len(attn_list)
    fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3))

    for layer in range(n_layers):
        ax = axes[layer]
        attn_matrix = attn_list[layer][0, head].cpu().numpy()
        sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens,
                    cmap="viridis", cbar=False, ax=ax)
        ax.set_title(f"{title_prefix}\nLayer {layer + 1}, Head {head + 1}")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = {
        "initial": "checkpoints/ckpt_init.pt",
        "middle": "checkpoints/ckpt_mid.pt",
        "final": "checkpoints/ckpt_final.pt"
    }

    tokenizer = SimpleMathTokenizer()
    sentence = "12+7="

    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found, skipping...")
            continue

        print(f"Loading {name} checkpoint from {ckpt_path}...")
        model = load_model(ckpt_path, device)

        fig = visualize_all_layers(
            model, tokenizer, sentence, device,
            head=0, title_prefix=f"{name} checkpoint"
        )

        os.makedirs("figures", exist_ok=True)
        out_path = f"figures/attention-{name}.png"
        fig.savefig(out_path)
        print(f"Saved attention visualization to {out_path}")
        plt.close(fig)



# From the beginning to the end of training, you can really see how the model’s attention changes. At first, it’s very simple—the attention just follows the diagonal, mostly looking at each token and maybe the one right next to it. In the middle stage, it starts to spread out, connecting digits with operators, which shows the model is beginning to pick up on the structure of the equation. By the final stage, the attention looks much more organized and global
