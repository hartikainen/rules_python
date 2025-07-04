"""A simple Jax/MuJoCo example."""

import dataclasses
import operator
import time
import textwrap

from absl import app
from absl import logging
from etils import eapp
from etils import epath
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np


@dataclasses.dataclass
class Args: ...


def main(args: Args) -> None:
    print(f"{args=}")

    x = jax.numpy.arange(3) + 1
    print(f"{(x, x.device)=}")

    model = mujoco.MjModel.from_xml_string(textwrap.dedent("""
      <mujoco>
        <worldbody>
          <body name="box" pos="0 0 1">
            <joint name="box" type="hinge"/>
            <geom type="box" size="0.5 0.5 0.5"/>
          </body>
        </worldbody>
      </mujoco>
    """))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    modelx = mjx.put_model(model)
    datax = mjx.put_data(model, data)

    key = jax.random.PRNGKey(seed=0)

    data.xfrc_applied[:] = jax.random.uniform(key, data.xfrc_applied.shape)
    datax = datax.replace(xfrc_applied=data.xfrc_applied)

    np.testing.assert_equal(data.qpos, 0.0)
    np.testing.assert_equal(np.array(datax.qpos), 0.0)

    mujoco.mj_step(model, data)
    datax = mjx.step(modelx, datax)

    # Make sure array has changed to non-zero.
    np.testing.assert_array_compare(operator.ne, data.qpos, 0.0)
    np.testing.assert_array_compare(operator.ne, datax.qpos, 0.0)
    np.testing.assert_allclose(data.qpos, datax.qpos, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    eapp.better_logging()
    app.run(main, flags_parser=eapp.make_flags_parser(Args))
