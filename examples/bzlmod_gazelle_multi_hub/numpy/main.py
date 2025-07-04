"""A simple script that consumes `numpy`."""

import dataclasses

from absl import app
from absl import logging
from etils import eapp
from etils import epath
import numpy as np


@dataclasses.dataclass
class Args: ...


def main(args: Args) -> None:
    print(f"{args=}")

    x = np.arange(3) + 1
    print(f"{x=}")


if __name__ == "__main__":
    eapp.better_logging()
    app.run(main, flags_parser=eapp.make_flags_parser(Args))
