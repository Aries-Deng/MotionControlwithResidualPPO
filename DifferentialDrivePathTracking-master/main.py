from pathlib import Path
import os

_CACHE_DIR = Path(__file__).resolve().parent / "outputs" / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_CACHE_DIR / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np


class State():
    def __init__(self, x_, y_, theta_):
        if not (x_ is None or y_ is None or theta_ is None):
            self.x = x_
            self.y = y_
            self.theta = theta_
        else:
            self.x = 0
            self.y = 0
            self.theta = 0

    def __str__(self):
        return str(self.x)+","+str(self.y)+","+str(self.theta)


class TimedState(State):
    def __init__(self, t_, x_, y_, theta_, speed_=0.0):
        super().__init__(x_, y_, theta_)
        self.t = t_
        self.speed = speed_


class Controller():
    def __init__(
            self, start_, goal_, R_=0.0325, L_=0.1,
            kP=1.0, kI=0.01, kD=0.01, dT=0.1, v=1.0,
            arrive_distance=1):

        self.current = start_
        self.goal = goal_
        self.R = R_  # in meter
        self.L = L_  # in meter

        self.E = 0   # Cummulative error
        self.old_e = 0  # Previous error

        self.Kp = kP
        self.Ki = kI
        self.Kd = kD

        self.desiredV = v
        self.dt = dT  # in second
        self.arrive_distance = arrive_distance
        return

    def uniToDiff(self, v, w):
        vR = (2*v + w*self.L)/(2*self.R)
        vL = (2*v - w*self.L)/(2*self.R)
        return vR, vL

    def diffToUni(self, vR, vL):
        v = self.R/2*(vR+vL)
        w = self.R/self.L*(vR-vL)
        return v, w

    def iteratePID(self):
        # Difference in x and y
        d_x = self.goal.x - self.current.x
        d_y = self.goal.y - self.current.y

        # Angle from robot to goal
        g_theta = np.arctan2(d_y, d_x)

        # Error between the goal angle and robot angle
        alpha = g_theta - self.current.theta
        # alpha = g_theta - math.radians(90)
        e = np.arctan2(np.sin(alpha), np.cos(alpha))

        e_P = e
        e_I = self.E + e
        e_D = e - self.old_e

        # This PID controller only calculates the angular
        # velocity with constant speed of v
        # The value of v can be specified by giving in parameter or
        # using the pre-defined value defined above.
        w = self.Kp*e_P + self.Ki*e_I + self.Kd*e_D

        w = np.arctan2(np.sin(w), np.cos(w))

        self.E = self.E + e
        self.old_e = e
        v = self.desiredV

        return v, w

    def fixAngle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def makeAction(self, v, w):
        x_dt = v*np.cos(self.current.theta)
        y_dt = v*np.sin(self.current.theta)
        theta_dt = w

        self.current.x = self.current.x + x_dt * self.dt
        self.current.y = self.current.y + y_dt * self.dt
        self.current.theta = self.fixAngle(
            self.current.theta + self.fixAngle(theta_dt * self.dt))
        return

    def isArrived(self):
        # print("Arrive check:", str(abs(self.current.x - self.goal.x)),
        #       str(abs(self.current.y - self.goal.y)))
        current_state = np.array([self.current.x, self.current.y])
        goal_state = np.array([self.goal.x, self.goal.y])
        difference = current_state - goal_state

        distance_err = difference @ difference.T
        if distance_err < self.arrive_distance:
            return True
        else:
            return False

    def runPID(self, parkour=None):
        x = [self.current.x]
        y = [self.current.y]
        theta = [self.current.theta]
        while(not self.isArrived()):
            v, w = self.iteratePID()
            self.makeAction(v, w)
            x.append(self.current.x)
            y.append(self.current.y)
            theta.append(self.current.theta)
            if parkour:

                parkour.drawPlot(x, y, theta)
            # time.sleep(self.dt)

            # Print or plot some things in here
            # Also it can be needed to add some max iteration for
            # error situations and make the code stable.
            # print(self.current.x, self.current.y, self.current.theta)
        return x, y, theta


def fixAngle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def make_time_indexed_trajectory(keyframes, dt):
    """Sample explicit-time keyframes into a fixed-frame reference trajectory."""
    if len(keyframes) < 2:
        raise ValueError("At least two keyframes are required.")

    reference = []
    for start, end in zip(keyframes[:-1], keyframes[1:]):
        if end.t <= start.t:
            raise ValueError("Keyframe timestamps must be strictly increasing.")

        dx = end.x - start.x
        dy = end.y - start.y
        duration = end.t - start.t
        segment_theta = np.arctan2(dy, dx)
        segment_speed = np.hypot(dx, dy) / duration
        segment_steps = int(np.ceil((end.t - start.t) / dt))
        for step in range(segment_steps):
            t = start.t + step * dt
            if reference and t <= reference[-1].t:
                continue
            ratio = min((t - start.t) / duration, 1.0)
            reference.append(
                TimedState(
                    t,
                    start.x + dx * ratio,
                    start.y + dy * ratio,
                    segment_theta,
                    segment_speed,
                )
            )

    final = keyframes[-1]
    reference.append(TimedState(final.t, final.x, final.y, reference[-1].theta, 0.0))
    return reference


def pose_tracking_error(actual, reference):
    position_error = np.hypot(actual.x - reference.x, actual.y - reference.y)
    heading_error = abs(fixAngle(actual.theta - reference.theta))
    pose_mae = np.mean(
        np.abs([
            actual.x - reference.x,
            actual.y - reference.y,
            fixAngle(actual.theta - reference.theta),
        ])
    )
    return position_error, heading_error, pose_mae


def trackTimedRoute(
        start,
        reference,
        dt=0.1,
        kP=1.5,
        kI=0.0,
        kD=0.5,
        speed_scale=1.0,
        lookahead_steps=10):
    current = State(start.x, start.y, start.theta)
    controller = Controller(current, reference[0], kP=kP, kI=kI, kD=kD, dT=dt)

    t = []
    x = []
    y = []
    theta = []
    ref_x = []
    ref_y = []
    ref_theta = []
    position_error = []
    heading_error = []
    pose_mae = []

    for index, target in enumerate(reference):
        controller.goal = reference[min(index + lookahead_steps, len(reference) - 1)]
        controller.desiredV = target.speed * speed_scale
        pos_err, head_err, mae = pose_tracking_error(controller.current, target)

        t.append(target.t)
        x.append(controller.current.x)
        y.append(controller.current.y)
        theta.append(controller.current.theta)
        ref_x.append(target.x)
        ref_y.append(target.y)
        ref_theta.append(target.theta)
        position_error.append(pos_err)
        heading_error.append(head_err)
        pose_mae.append(mae)

        v_cmd, w_cmd = controller.iteratePID()
        controller.makeAction(v_cmd, w_cmd)

    return {
        "t": np.asarray(t),
        "x": np.asarray(x),
        "y": np.asarray(y),
        "theta": np.asarray(theta),
        "ref_x": np.asarray(ref_x),
        "ref_y": np.asarray(ref_y),
        "ref_theta": np.asarray(ref_theta),
        "position_error": np.asarray(position_error),
        "heading_error": np.asarray(heading_error),
        "pose_mae": np.asarray(pose_mae),
    }


def save_tracking_plots(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_position_error = float(np.mean(history["position_error"]))
    mean_heading_error = float(np.mean(history["heading_error"]))
    mean_pose_mae = float(np.mean(history["pose_mae"]))

    plt.figure(figsize=(9, 5))
    plt.plot(history["t"], history["position_error"], label="position error [m]")
    plt.plot(history["t"], history["heading_error"], label="heading error [rad]")
    plt.axhline(0.0, color="r", linestyle="--", label="desired")
    plt.title(
        "Differential-drive tracking error: "
        f"pos={mean_position_error:.3f} m, heading={mean_heading_error:.3f} rad"
    )
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    error_plot = output_dir / "differential_drive_tracking_error.png"
    plt.savefig(error_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(history["ref_x"], history["ref_y"], "k--", label="time-indexed reference")
    plt.plot(history["x"], history["y"], "b", label="PID rollout")
    plt.scatter(history["ref_x"][0], history["ref_y"][0], c="g", label="start")
    plt.scatter(history["ref_x"][-1], history["ref_y"][-1], c="r", label="end")
    plt.title(f"Differential-drive path tracking, pose MAE={mean_pose_mae:.3f}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path_plot = output_dir / "differential_drive_tracking_path.png"
    plt.savefig(path_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history["t"], np.unwrap(history["ref_theta"]), "k--", label="reference heading")
    plt.plot(history["t"], np.unwrap(history["theta"]), "b", label="PID heading")
    plt.title(f"Differential-drive heading tracking, mean error={mean_heading_error:.3f} rad")
    plt.xlabel("time [s]")
    plt.ylabel("heading [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    heading_plot = output_dir / "differential_drive_heading_tracking.png"
    plt.savefig(heading_plot, dpi=160)
    plt.close()

    return {
        "mean_position_error": mean_position_error,
        "mean_heading_error": mean_heading_error,
        "mean_pose_mae": mean_pose_mae,
        "error_plot": error_plot,
        "path_plot": path_plot,
        "heading_plot": heading_plot,
    }


def challenge_keyframes():
    """Return the shared time-indexed challenge route used by PID and PPO."""
    return [
        TimedState(0.0, -20.0, 15.0, 0.0),
        TimedState(6.5, -14.0, 22.0, 0.0),
        TimedState(13.0, -8.0, 2.0, 0.0),
        TimedState(21.0, 0.0, 21.0, 0.0),
        TimedState(29.0, 8.0, -2.0, 0.0),
        TimedState(38.5, 14.0, 16.0, 0.0),
        TimedState(48.0, 22.0, -6.0, 0.0),
    ]


# Starting point of the code
def main():
    dt = 0.1
    start = State(-20.0, 15.0, np.radians(90))

    # Explicit-time challenge reference: each keyframe says where the car should
    # be and when it should be there. The short zigzags, reversals, and speed
    # changes make a simple heading-only PID visibly work harder than before.
    reference = make_time_indexed_trajectory(challenge_keyframes(), dt)
    history = trackTimedRoute(
        start,
        reference,
        dt=dt,
        kP=2.0,
        kI=0.005,
        kD=0.25,
        speed_scale=0.985,
        lookahead_steps=9,
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    summary = save_tracking_plots(history, output_dir)
    print(
        "Differential-drive tracking error: "
        f"position={summary['mean_position_error']:.3f} m, "
        f"heading={summary['mean_heading_error']:.3f} rad, "
        f"pose_mae={summary['mean_pose_mae']:.3f}"
    )
    print(f"Saved error plot: {summary['error_plot']}")
    print(f"Saved path plot: {summary['path_plot']}")
    print(f"Saved heading plot: {summary['heading_plot']}")


if __name__ == "__main__":
    main()
