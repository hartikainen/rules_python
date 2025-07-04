"""A HTTP server example."""

from collections.abc import Sequence
import dataclasses
import os
import signal
import time

from absl import app
from absl import logging
from etils import eapp
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np

from flask import Flask

server = Flask(__name__)


@dataclasses.dataclass
class Args: ...


@server.route("/")
def index() -> str:
    return f"Hello, World! This is Flask."


def main(argv: Sequence[str]) -> None:
    del argv

    def handler(signum, frame):
        # Raising an exception is a clean way to break out of server.run()
        raise SystemExit("Server timed out, shutting down.")

    # Set the alarm signal handler
    signal.signal(signal.SIGALRM, handler)
    # Trigger the alarm after 5 seconds
    signal.alarm(2)

    x = jax.numpy.arange(3) + 1
    print(x, np.mean(x))

    try:
        server.run(host="0.0.0.0", port=8080, debug=False)
    except SystemExit as e:
        print(e)


if __name__ == "__main__":
    eapp.better_logging()
    app.run(main, flags_parser=eapp.make_flags_parser(Args))
