# discrete_controller.py

class DiscreteController:
    def __init__(self):
        # Action ID → (vx, vy, yaw_rate, dz)
        self.actions = {
            0: (0.03, 0.0, 0.0, 0.0),   # Forward
            1: (-0.03, 0.0, 0.0, 0.0),  # Backward
            2: (0.0, 0.03, 0.0, 0.0),   # Right
            3: (0.0, -0.03, 0.0, 0.0),  # Left
            4: (0.0, 0.0, +0.4, 0.0),  # Yaw right
            5: (0.0, 0.0, -0.4, 0.0),  # Yaw left
            6: (0.0, 0.0, 0.0, +0.5),  # Up
            7: (0.0, 0.0, 0.0, -0.5),  # Down
            8: (0.0, 0.0, 0.0, 0.0),     # Hover

            # New systematic search actions
            9: (0.03, 0.0, 0.1, 0.0),  # Spiral right (forward + yaw)
            10: (0.03, 0.0, -0.1, 0.0),  # Spiral left

            11: (0.0, 0.0, 1.57, 0.0),
            12: (0.0, 0.0, -1.57, 0.0)
        }

        self.forward   = 0
        self.backward  = 1
        self.left      = 2
        self.right     = 3
        self.yaw_left  = 4
        self.yaw_right = 5
        self.ascend    = 6
        self.descend   = 7
        self.hover     = 8

        self.spright = 9
        self.spleft = 10

        self.halfspinright = 11
        self.halfspinleft = 12

    def get_command(self, action_id: int):
        if action_id not in self.actions:
            raise ValueError(f"Invalid action ID: {action_id}")
        return self.actions[action_id]  # returns (vx, vy, yaw_rate, dz)
