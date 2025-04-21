# -------------- webots_drone_api.py --------------
from controller import Robot, Camera, GPS, DistanceSensor, Supervisor
import numpy as np, math
from pid_controller import pid_velocity_fixed_height_controller
import random

class WebotsDroneAPI:
    _MOTOR_SIGNS = (-1, +1, -1, +1)
    def __init__(self, timestep=32, z_max=2.0, epsilon=0.1):
        self.robot = Supervisor()
        self.timestep_ms = int(timestep)
        self.z_max = float(z_max)

        # Devices initialization
        self._motors = self._init_motors()
        self.gps = self._require("gps")
        self.camera = self._optional("camera", enable=True)
        if self.camera is not None:
            self.camera.enable(self.timestep_ms)
        self.gps.enable(self.timestep_ms)
        self.imu = self._optional("inertial_unit", enable=True)
        self.gyro = self._optional("gyro", enable=True)
        self.range = self._optional("range_front", enable=True)
        self.ds = [self._optional(n) for n in [
            "front_ds", "back_ds", "left_ds", "right_ds"]]
        for s in self.ds:
            s.enable(self.timestep_ms)

        # State initialization
        self.pid = pid_velocity_fixed_height_controller()
        self._t_last = self.robot.getTime()
        self.z_target = 0.3  # start 30 cm above ground
        self._last_state = {"vx": 0, "vy": 0}
        self._valid_altitude = False



        print("[INIT] WebotsDroneAPI ready — timestep %d ms" % self.timestep_ms)

    def get_camera_image(self):
        if self.camera:
            img = self.camera.getImage()
            w, h = self.camera.getWidth(), self.camera.getHeight()
            # Convert Webots image to numpy
            return np.frombuffer(img, np.uint8).reshape((h, w, 4))[:, :, :3]  # RGB
        return None

    def step(self):
        """
        This method advances the simulation by one timestep.
        It calls the `step` method of the Webots Robot class.
        """
        self.robot.step(self.timestep_ms)

    def _init_motors(self):
        """
        Initialize motors and return the motor objects.
        Adjust this method according to the specific robot configuration.
        """
        motor_names = ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]
        motors = []

        for motor_name in motor_names:
            motor = self._require(motor_name)
            if motor is None:
                print(f"[ERROR] Motor {motor_name} not found. Please check your Webots configuration.")
                continue
            motor.setPosition(float("inf"))  # Set motors to velocity control mode
            motor.setVelocity(0)  # Set initial motor velocity to 0
            motors.append(motor)

        # Check if all motors are initialized correctly
        if len(motors) != 4:
            raise ValueError("[ERROR] Not all motors initialized. Please check the motor configuration.")

        return motors

    def _require(self, device_name):
        """
        Helper function to retrieve and initialize a device in Webots.
        This will return None if the device is not found.
        """
        try:
            device = self.robot.getDevice(device_name)
            return device
        except Exception as e:
            print(f"[ERROR] Device {device_name} could not be found: {e}")
            return None

    def _optional(self, device_name, enable=False):
        """
        Initialize optional devices, like sensors.
        """
        device = self._require(device_name)
        if device and enable:
            device.enable(self.timestep_ms)
        return device

    def get_state(self):
        x, y, z = self.gps.getValues()

        # Calculate velocities
        if not hasattr(self, "_prev_gps"):
            self._prev_gps = (x, y, z)
            self._prev_time = self.robot.getTime()
            vx = vy = vz = 0.0
        else:
            dt = max(self.robot.getTime() - self._prev_time, 1e-6)
            vx = (x - self._prev_gps[0]) / dt
            vy = (y - self._prev_gps[1]) / dt
            vz = (z - self._prev_gps[2]) / dt
            self._prev_gps = (x, y, z)
            self._prev_time = self.robot.getTime()

        # Get roll, pitch, yaw
        roll = pitch = yaw = 0.0
        if self.imu:
            roll, pitch, yaw = self.imu.getRollPitchYaw()

        # Front distance
        #front_dist = self.range.getValue() / 1000.0 if self.range else 999.0
        def safe_get(sensor):
            return sensor.getValue() / 1000.0 if sensor else 999.0

        return {
            "x": x, "y": y, "z": z,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "vx": vx, "vy": vy, "vz": vz,
            "front_dist": safe_get(self.ds[0]),
            "back_dist": safe_get(self.ds[1]),
            "left_dist": safe_get(self.ds[2]),
            "right_dist": safe_get(self.ds[3])
        }

    def send_command(self, vx: float, vy: float, yaw_rate: float, dz: float):
        x, y, z = self.gps.getValues()
        if np.isnan(z):
            print("[WAIT] GPS altitude NaN — skipping control step")
            return
        if not self._valid_altitude:
            self._valid_altitude = True
            self.z_target = z + 0.3  # lift off a bit
            print(f"[INIT] Altitude OK → z_target = {self.z_target:.2f} m")

        t_now = self.robot.getTime()
        dt = max(t_now - self._t_last, 1e-6)
        self._t_last = t_now

        # Update target height with dz (change in altitude)
        dz = np.clip(dz, -2.0, 2.0)
        self.z_target = np.clip(self.z_target + dz * dt, 0.1, self.z_max)

        # orientation and sensor readings
        roll = pitch = yaw_rate_meas = 0.0
        if self.imu:
            roll, pitch, _ = self.imu.getRollPitchYaw()
        if self.gyro:
            yaw_rate_meas = self.gyro.getValues()[2]

        # Get measured horizontal velocities (from velocity estimator)
        vx_meas = self._last_state.get("vx", 0.0)
        vy_meas = self._last_state.get("vy", 0.0)

        # ----- RUN PID ----- #
        motor_cmd = self.pid.pid(
            dt, vx, vy, yaw_rate, self.z_target,
            roll, pitch, yaw_rate_meas,
            z, vx_meas, vy_meas
        )

        motor_vel = [np.clip(cmd * 2.0, 0.0, 1000.0) for cmd in motor_cmd]
        for m, sign, vel in zip(self._motors, self._MOTOR_SIGNS, motor_vel):
            m.setVelocity(sign * vel)

        # Store estimated vx, vy
        self._last_state["vx"] = vx_meas
        self._last_state["vy"] = vy_meas

        # ----- debug print ----- #
        print(
            f"[PID] dt={dt * 1000:5.1f} ms  z={z:5.2f}→{self.z_target:4.2f} m  "
            f"vx={vx:.2f} vy={vy:.2f}  motors={[round(sign * v, 1) for sign, v in zip(self._MOTOR_SIGNS, motor_vel)]}"
        )

    def epsilon_greedy_action(self):
        """
        Epsilon-Greedy action selection.
        With probability epsilon, explore randomly.
        Otherwise, choose the best action based on the policy.
        """
        # Random exploration (epsilon probability)
        if random.random() < self.epsilon:  # Explore randomly
            # Define the range of possible actions, e.g., actions 1-5
            return random.choice([0, 1, 2, 3, 4, 5])  # Modify based on your action space
        else:  # Exploit learned policy (choose best action)
            # Choose the action based on a policy (e.g., from a trained model or a heuristic)
            return self.best_action_based_on_policy()

    def best_action_based_on_policy(self):
        """
        Return the best action based on the current policy. This can be based on a
        pre-trained model or a heuristic.
        """
        # This function should return the action based on your trained model/policy
        return 0  # Modify this according to your model

    def pid_controller(self, z, dz):
        """
        Refactored PID controller for altitude stability.
        """
        # Calculate the altitude error
        err = self.z_target - z  # Error in altitude
        # Compute motor thrust based on PID
        thrust = self._hover + 400 * err - 70 * dz  # Proportional-Derivative control
        thrust = np.clip(thrust, 0, 1000)  # Limit thrust to prevent overshooting
        return thrust

    # In webots_drone_api.py, add this method to the WebotsDroneAPI class
    def reset(self):
        """Reinitialize motors and sensors after simulation reset."""
        self._motors = self._init_motors()  # Reinitialize motors in velocity mode
        self.gps.enable(self.timestep_ms)
        if self.camera:
            self.camera.enable(self.timestep_ms)
        # Re-enable other sensors if needed
        print("[RESET] Motors and sensors reinitialized.")
# ------------------------------------------------ END FILE ----------
