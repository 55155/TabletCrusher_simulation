import genesis as gs
import pytest
import torch
import torch
import numpy as np
from genesis.utils.misc import tensor_to_array



def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=""):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    if isinstance(actual, torch.Tensor):
        actual = tensor_to_array(actual)
    actual = np.asanyarray(actual)
    if isinstance(desired, torch.Tensor):
        desired = tensor_to_array(desired)
    desired = np.asanyarray(desired)

    if all(e.size == 0 for e in (actual, desired)):
        return

    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol, err_msg=err_msg)


def test_urdf_mimic(show_viewer, tol):
    # create and build the scene
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    hand = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/hand.urdf",
            fixed=True,
        ),
    )
    scene.build()
    assert scene.rigid_solver.n_equalities == 1

    qvel = scene.rigid_solver.dofs_state.vel.to_numpy()
    qvel[-1] = 1
    scene.rigid_solver.dofs_state.vel.from_numpy(qvel)
    for i in range(200):
        scene.step()

    gs_qpos = scene.rigid_solver.qpos.to_numpy()[:, 0]
    assert_allclose(gs_qpos[-1], gs_qpos[-2], tol=tol)

if __name__ == "__main__":
    gs.init(backend=gs.cuda)
    test_urdf_mimic(show_viewer=True, tol=1e-3)