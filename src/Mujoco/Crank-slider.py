import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

SCENE_XML = "/home/seongjin/Desktop/Seongjin/genesis_simulation_on_linux/My_asset/Scene_description/Scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE_XML)  # [web:2]
data  = mujoco.MjData(model)                     # [web:2]

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)  # [web:53]
else:
    mujoco.mj_resetData(model, data)             # [web:53]
mujoco.mj_forward(model, data)                   # [web:53]

crank_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "crank_motor")  # [web:2]
tablet_bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablet")           # [web:2]

# ---- plot ----
plt.ion()  # [web:242]
fig, ax = plt.subplots()
ax.set_title("Tablet force components")
ax.set_xlabel("time [s]")
ax.set_ylabel("Force [N] (approx)")
ax.grid(True)

N = 400
ts  = deque(maxlen=N)
fxs = deque(maxlen=N)
fys = deque(maxlen=N)
fzs = deque(maxlen=N)

(line_fx,) = ax.plot([], [], label="Fx")
(line_fy,) = ax.plot([], [], label="Fy")
(line_fz,) = ax.plot([], [], label="Fz")
ax.legend(loc="upper right")

def update_plot():
    line_fx.set_data(ts, fxs)
    line_fy.set_data(ts, fys)
    line_fz.set_data(ts, fzs)

    if len(ts) > 2:
        ax.set_xlim(ts[0], ts[-1])
        ymin = min(min(fxs), min(fys), min(fzs))
        ymax = max(max(fxs), max(fys), max(fzs))
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0
        ax.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))

    fig.canvas.draw_idle()
    plt.pause(0.001)  # 이벤트 루프 처리 [web:260]

PLOT_EVERY = 10   # <- 여기 값을 5~10으로 조절

with mujoco.viewer.launch_passive(model, data) as viewer:  # [web:2]
    step_count = 0
    while viewer.is_running():
        data.ctrl[crank_act_id] = 3.0  # [web:4]

        mujoco.mj_step(model, data)                 # [web:36]
        mujoco.mj_rnePostConstraint(model, data)    # [web:53]

        wrench = data.cfrc_ext[tablet_bid].copy()   # [web:53]
        f = wrench[3:6]                             # [web:228]

        # 데이터는 매 step 저장(원하면 이것도 줄일 수 있음)
        ts.append(float(data.time))
        fxs.append(float(f[0]))
        fys.append(float(f[1]))
        fzs.append(float(f[2]))

        # 플랏만 N step마다 갱신
        if step_count % PLOT_EVERY == 0:
            update_plot()

        viewer.sync()  # [web:2]
        time.sleep(model.opt.timestep)

        step_count += 1

plt.ioff()
plt.show()
