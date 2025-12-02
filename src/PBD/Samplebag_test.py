import genesis as gs

########################## init ##########################
gs.init()

########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt       = 0.01,
        substeps = 10,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov = 30,
        res        = (1280, 720),
        max_FPS    = 60,
    ),
    show_viewer = True,
)
Samplebag_filename = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Sample_bag.obj"

########################## entities ##########################
rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
    material=rigid_material,
)

Samplebag = scene.add_entity(
    material=gs.materials.FEM.Cloth(),
    morph=gs.morphs.Mesh(
        file = Samplebag_filename,
        scale=0.001,
    ),
    surface=gs.surfaces.Default(
        smooth=True,
        color=(1.0, 1.0, 1.0, 1.0),
        opacity = 0.3,
        vis_mode='visual',
    )
)
cam = scene.add_camera(
    res=(1280, 960),
    pos=(0.5, 0.5, 0.5),
    lookat=(0.0500, 0.0500, 0.1000), 
) 

# tablet = scene.add_entity(
#     material=rigid_material,
#     morph=gs.morphs.MJCF(
#         file = '/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Tablet_posmod/Tablet_posmod.xml',
#         scale = 1.0, pos = (0,0,10), euler = (0,0,0), decimate = False, convexify = False,),
#     surface=gs.surfaces.Default(
#         smooth=False,
#     ),
#     vis_mode="collision",
#     visualize_contact=True,
# )

tablet = scene.add_entity(
    material=rigid_material,
    morph = gs.morphs.Mesh(
        file = '/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Tablet_posmod/Tablet_posmod.obj',
        scale = 0.002, pos = (0.0500, 0.0500, 3.1000), euler = (0,0,0),
    ),
    vis_mode="collision",
    visualize_contact=True,
)

scene.build()
cam.start_recording()

a = Samplebag.find_closest_particle((-1, -1, 1.0))
b = Samplebag.find_closest_particle((1, 1, 1.0)) 

print("좌측상단 : ", a)
print("우측상단 : ", b)
print("중간상단 :", (a+b)/2)

particle_pos = Samplebag.get_particles_pos()
left_top_pos = particle_pos[a]
right_top_pos = particle_pos[b]
center_top_pos = (left_top_pos + right_top_pos) / 2
print("중앙상단 좌표 :", center_top_pos)

# tablet.set_pos(center_top_pos)

Samplebag.fix_particles(Samplebag.find_closest_particle((-1, -1, 1.0)))
Samplebag.fix_particles(Samplebag.find_closest_particle((1, 1, 1.0)))

horizon = 400
for i in range(horizon):
    scene.step()
    # if i == 100:
    #     tablet.set_pos((0.0500, 0.0500, 0.0500))
    cam.render()
cam.stop_recording(save_to_filename="video/[20251202] Samplebag_test4.mp4")
