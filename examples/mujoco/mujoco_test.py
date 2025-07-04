"""A simple MuJoCo test."""

import mujoco
from mujoco import mjx


def main() -> None:
    """Run a simple MuJoCo test."""
    model = mujoco.MjModel.from_xml_string(
        """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="0 0 1"/>
        <body name="box" pos="0 0 1">
          <geom name="boxgeom" type="box" size="0.1 0.1 0.1" density="100"/>
          <joint name="boxfree" type="free"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )
    data = mujoco.MjData(model)
    modelx = mjx.put_model(model)
    datax = mjx.put_data(modelx, data)

    mujoco.mj_resetData(model, data)
    for _ in range(2):
        mujoco.mj_step(model, data)
    print("Box position:", data.xpos[1])


if __name__ == "__main__":
    main()
