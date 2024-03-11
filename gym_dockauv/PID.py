class PID_Controller:
    # 给pid的三个参数赋初值
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0.0
        self.integral = 0.0
        self.derivative = []

    def change_para(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def control_action(self, error, dt):
        """
        Args:
            error: 当前误差
            dt: 步长

        Returns: pid的输出
        """
        p = self.kp * (error)

        self.integral += error
        i = self.ki * self.integral

        derivative = (error - self.last_error) / dt
        d = self.kd * derivative
        self.last_error = error

        return p + i + d

