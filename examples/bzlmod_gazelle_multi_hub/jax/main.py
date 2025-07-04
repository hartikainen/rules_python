"""A simple script that consumes `jax`."""

import dataclasses

from absl import app
from absl import logging
from etils import eapp
from etils import epath
import jax.numpy as jnp


@dataclasses.dataclass
class Args: ...


def main(args: Args) -> None:
    print(f"{args=}")

    x = jnp.arange(3) + 1
    print(f"{(x, x.device)=}")


if __name__ == "__main__":
    eapp.better_logging()
    app.run(main, flags_parser=eapp.make_flags_parser(Args))
