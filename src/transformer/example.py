import functools
import torch
import pandas as pd
import altair as alt


import warnings

from pathlib import Path
from rich.console import Console
from rich import print

from .transformer import subsequent_mask, PositinalEncoding, make_model

console = Console()


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
FIGURE_PATH = Path("figures/")
EXAMPLES = []


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    def step(self):
        pass


def save_char(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        char = func(*args, **kwargs)
        char.save(FIGURE_PATH / f"{func.__name__}.html")
        return char

    return wrapper


def example(func):
    EXAMPLES.append(func.__name__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if RUN_EXAMPLES:
            console.print(f"Running Example: {func.__name__}", style="bold red")
            return func(*args, **kwargs)

    return wrapper


@example
@save_char
def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    char = (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )

    return char


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
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    char = (
        alt.Chart(data)
        .mark_line()
        .properties(width=1000)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )

    return char


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type(torch.long)

    for i in range(9):
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)))
        prob = test_model.generator(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print(f"Example Untrained Model Predition: {ys}")


@example
def run_test():
    for _ in range(10):
        inference_test()


def main():
    console.print(f"Examples: {EXAMPLES}\n")
    run_test()
