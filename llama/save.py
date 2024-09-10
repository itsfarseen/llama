import json
import time
from pathlib import Path
from pprint import pprint

import torch
import torch.ao.quantization as torchquant
import torch.distributed as dist

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

from fairscale.nn.model_parallel.initialize import initialize_model_parallel


OUTPUT = "llama-2-7b-int8/quantized-llama-2-7b.pth"
CKPT_DIR = "./llama-2-7b/"
TOKENIZER_PATH = "tokenizer.model"
MAX_SEQ_LEN = 128
MAX_BATCH_SIZE = 4
PARAMS = {}


class LogTime:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        print(self.name)
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed = time.time() - self.start_time  # type: ignore
        print(f"  Done in {elapsed:.2f} seconds")


class TorchDistributed:
    def __init__(self):
        pass

    def __enter__(self):
        with LogTime("Initializing torch distributed"):
            backend = "gloo"
            init_method = "tcp://127.0.0.1:29500"
            world_size = 1
            rank = 0
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
            dist.new_group()
            if dist.is_initialized():
                print("Distributed initialization successful")
            else:
                print("Failed to initialize distributed process group")
            initialize_model_parallel(1)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        dist.destroy_process_group()


def main():
    with TorchDistributed():
        quantize_model()


def quantize_model():
    torch.set_default_dtype(torch.float16)
    with LogTime("Loading checkpoint"):
        checkpoint = load_checkpoint()

    model_args = get_model_args()
    model = Transformer(model_args)

    with LogTime("Loading state dict"):
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    inspect_model_size(model)

    model.qconfig = torchquant.default_qconfig  # type: ignore

    with LogTime("Preparing for quantization"):
        torchquant.prepare(model, inplace=True)

    # calibrate?
    # with torch.no_grad():
    #     model(input, 0)

    with LogTime("Quantizing"):
        torchquant.convert(model, inplace=True)

    inspect_model_size(model)

    with LogTime("Saving"):
        save_checkpoint(model)


def generate_input_vector():
    input_shape = (MAX_BATCH_SIZE, MAX_SEQ_LEN)
    input = torch.zeros(input_shape, dtype=torch.long)
    return input


# Helpers


def prod(x):
    p = 1
    for i in x:
        p *= i
    return p


def inspect_model_size(model: Transformer):
    model_state_dict = model.state_dict()
    size = 0
    dtypes = set()
    for i, (key, value) in enumerate(model_state_dict.items()):
        # pprint((i, key, value.dtype, value.shape))
        dtypes.add(value.dtype)
        size += prod(value.shape)
    print("Params in B", size / (1024 * 1024 * 1024))
    print("Layer dtypes", dtypes)


def load_checkpoint():
    checkpoints = sorted(Path(CKPT_DIR).glob("*.pth"))

    assert len(checkpoints) > 0, f"no checkpoint files found in {CKPT_DIR}"
    ckpt_path = checkpoints[0]

    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True)
    return checkpoint


def get_model_args():
    with open(Path(CKPT_DIR) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        **params,
    )
    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)
    model_args.vocab_size = tokenizer.n_words
    return model_args


def save_checkpoint(model: Transformer):
    torch.save(model.state_dict(), OUTPUT)


if __name__ == "__main__":
    main()
