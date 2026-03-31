import argparse
import os
import numpy as np  
from tqdm import tqdm

import genesis as gs
from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE, IS_PYQTGRAPH_AVAILABLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--timestep", type=float, default=0.01, help="Simulation time step")
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Show visualization GUI")
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-t", "--seconds", type=float, default=2.0, help="Number of seconds to simulate")
    parser.add_argument("-f", "--force", action="store_true", default=True, help="Use ContactForceSensor (xyz float)")
    parser.add_argument("-nf", "--no-force", action="store_false", dest="force", help="Use ContactSensor (boolean)")

    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level=None)

    ########################## scene setup ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.timestep,
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_timeconst=max(0.01, 2 * args.timestep),
            use_gjk_collision=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )
    cam = scene.add_camera(
        res=(1280, 960),
        pos = (3,0,3),
        lookat=(0, 0, 0.0),
        fov=30,
        
        GUI=True,
    )

    # rigid solver : for add_constaraints
    solver = scene.sim.rigid_solver
    scene.add_entity(gs.morphs.Plane())

    tablet_link_name = ("Tablet", "segment")
    tablet = scene.add_entity(
        gs.morphs.MJCF(
            file = "My_asset/Tablet_posmod/Tablet_posmod.xml",
            pos = (0, 0, 5.0),
            scale = 10.0,
        )
    )
    
    for link_name in tablet_link_name:
        if args.force:
            sensor_options = gs.sensors.ContactForce(
                entity_idx=tablet.idx,
                link_idx_local=tablet.get_link(link_name).idx_local,
                draw_debug=True,
            )
            plot_kwargs = dict(
                title=f"{link_name} Force Sensor Data",
                labels=["force_x", "force_y", "force_z"],
            )
        else:
            sensor_options = gs.sensors.Contact(
                entity_idx=tablet.idx,
                link_idx_local=tablet.get_link(link_name).idx_local,
                draw_debug=True,
            )
            plot_kwargs = dict(
                title=f"{link_name} Contact Sensor Data",
                labels=["in_contact"],
            )
    sensor = scene.add_sensor(sensor_options)
    if IS_PYQTGRAPH_AVAILABLE:
        sensor.start_recording(gs.recorders.PyQtLinePlot(**plot_kwargs))
    elif IS_MATPLOTLIB_AVAILABLE:
        print("pyqtgraph not found, falling back to matplotlib.")
        sensor.start_recording(gs.recorders.MPLLinePlot(**plot_kwargs))
    else:
        print("matplotlib or pyqtgraph not found, skipping real-time plotting.")

    ## scene build
    scene.build()
    cam.start_recording()
    # link1 = tablet.get_link(tablet_link_name[0])
    # link2 = tablet.get_link(tablet_link_name[1])
    # link1_idx_arr = np.array(link1.idx, dtype=gs.np_int)
    # link2_idx_arr = np.array(link2.idx, dtype=gs.np_int)
    # solver.add_weld_constraint(link1_idx_arr, link2_idx_arr)

    try:
        steps = int(args.seconds / args.timestep) if "PYTEST_VERSION" not in os.environ else 10
        for _ in range(steps):
            print(sensor.read())
            cam.render()
            scene.step()
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
        cam.stop_recording(save_to_filename ="video/SensorTEST_V2_20251014.mp4")
        scene.stop_recording()


if __name__ == "__main__":
    main()