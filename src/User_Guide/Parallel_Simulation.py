import torch
import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer   = False,
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build(n_envs=3000)

# control all the robots
franka.control_dofs_position(
    torch.tile(
        # N
        torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0.02, 0.02], device=gs.device), (3000, 1)
    ),
)

for i in range(1000):
    scene.step()