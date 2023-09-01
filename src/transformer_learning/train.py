import time
from dataclasses import dataclass

from rich import print
from rich.console import Console

from .transformer import subsequent_mask

console = Console()


class Batch:
    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        return tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)


@dataclass
class TrainState:
    step: int = 0
    accum_step: float = 0.0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode in ("train", "train+log"):
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.size(0)
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1

            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and mode in ("train", "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start

            print(
                (
                    f"Epoch Step: {i:6} | Accum: {n_accum:3} | LR: {lr:6.2f} |"
                    f"Tokens / Sec {tokens/elapsed:7.1f} | Learning Rate: {lr:6.1f}",
                ),
            )

            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """

    .. math::
        factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
