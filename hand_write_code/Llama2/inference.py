from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLAMA:

    def __init__(self, model_args: ModelArgs, tokenizer: SentencePieceProcessor, model: Transformer):
        self.args = model_args
        self.tokenizer = tokenizer
        self.model = model

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_length: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints files found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f}")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_length,
            max_batch_size = max_batch_size,
            device = device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f}s")

        return LLAMA(model_args, tokenizer, model)


if __name__=="__main__":
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LLAMA.build(
        checkpoints_dir="./checkpoints",
        tokenizer_path="./tokenizer.model",
        load_model=True,
        max_seq_length=1024,
        max_batch_size=3,
        device=device
    )

    print("All OK")