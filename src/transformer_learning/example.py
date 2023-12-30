import functools
import ssl
import subprocess
import warnings
from pathlib import Path

import altair as alt
import pandas as pd
import torch
from rich import print
from rich.console import Console
from torch.optim.lr_scheduler import LambdaLR

from .train import (
    LabelSmoothing,
    SimpleLossCompute,
    data_gen,
    greedy_decode,
    load_tokenizers,
    load_vocab,
    loss,
    rate,
    run_epoch,
)
from .transformer import PositinalEncoding, device, make_model, subsequent_mask

ssl._create_default_https_context = ssl._create_unverified_context


console = Console()

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
FIGURE_PATH = Path("figures/")
EXAMPLES = []


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=None):
    if args is None:
        args = []
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)
    return None


def execute_example(fn, args=None):
    if args is None:
        args = []
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, *, set_to_none=False):
        pass


class DummyScheduler:
    def step(self):
        pass


def save_char(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        char = func(*args, **kwargs)
        char.save(FIGURE_PATH / f"{func.__name__}.html")
        subprocess.check_call(["open", str(FIGURE_PATH / f"{func.__name__}.html")])
        return char

    return wrapper


def example(func):
    EXAMPLES.append(func.__name__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if RUN_EXAMPLES:
            console.print(f"Running Example: {func.__name__}", style="bold red")
            return func(*args, **kwargs)
        return None

    return wrapper


@example
@save_char
def example_mask():
    ls_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                },
            )
            for y in range(20)
            for x in range(20)
        ],
    )

    return (
        alt.Chart(ls_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


@example
@save_char
def example_positional():
    pe = PositinalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                },
            )
            for dim in [4, 5, 6, 7]
        ],
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=1000)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


def inference_test():
    test_model = make_model(11, 11, 2)

    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device())
    src_mask = torch.ones(1, 1, 10, device=device())

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1, device=device()).type(torch.long)

    for _ in range(9):
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)))
        prob = test_model.generator(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)],
            dim=1,
        )

    print(f"Example Untrained Model Predition: {ys}")


@example
def run_test():
    for _ in range(10):
        inference_test()


@example
@save_char
def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for _, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(),
            lr=1,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, *example),
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for _ in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                },
            )
            for warmup_idx in [0, 1, 2]
        ],
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=800)
        .encode(x="step", y="Learning Rate", color=r"model_size\:warmup:N")
        .interactive()
    )


@example
@save_char
def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ],
    )

    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))

    assert crit.true_dist is not None  # noqa: S101

    ls_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                },
            )
            for y in range(5)
            for x in range(5)
        ],
    )

    return (
        alt.Chart(ls_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=250, width=250)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color("target distribution:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


@example
@save_char
def penalization_visulization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Step": list(range(99)),
        },
    ).astype(float)

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Step",
            y="Loss",
        )
        .interactive()
    )


@example
def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    console.print(f"Using {device()}", style="bold green")
    model.to(device())

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.5,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    lr_scheduler = LambdaLR(
        optimizer,
        lambda step: rate(
            step,
            model_size=model.src_embed[0].d_model,
            factor=1.0,
            warmup=400,
        ),
    )

    batch_size = 80

    for _ in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device())
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len, device=device())
    print(greedy_decode(model, src, src_mask, max_len, start_symbol=0))


def main():
    console.print(f"Examples: {EXAMPLES}\n")
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
