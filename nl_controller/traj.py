import numpy as np

class TargetTraj():
    def __init__(self, FLAG = 1, h_default = 5.0):
        self.FLAG = FLAG    #选择轨迹类型，1：xy平面圆轨迹 2:平面8字（里萨如图形） 4:小圆轨迹(R=2.5m,T=20s) 5:直线运动(-4,0,5)到(4,0,5)T=16s 6:李萨如曲线(x:2.5sin(wt),y:2.5sin(2wt),T=25s) 7:z轴运动(加速度az=0.1*t)
        self.R = 8.0        #圆轨迹半径
        self.w = (2*np.pi)/30#轨迹周期
        self.h_default = h_default
        # FLAG=4 的参数：小圆轨迹
        self.R4 = 2.5       # 半径2.5米
        self.w4 = (2*np.pi)/20  # 20秒一圈
        # FLAG=5 的参数：直线运动
        self.T5 = 8.0      # 总时间16秒
        self.x_start = 0.0 # 起始x坐标
        self.x_end = 8.0    # 终止x坐标
        self.y5 = 0.0       # y坐标固定
        self.z5 = 5.0       # z坐标固定
        self.vx5 = (self.x_end - self.x_start) / self.T5  # 匀速速度
        # FLAG=6 的参数：李萨如曲线（x: 2.5sin(wt), y: 2.5sin(2wt)）
        self.A6 = 2.5       # 幅度2.5米
        self.T6 = 25.0      # x方向周期25秒
        self.w6 = (2*np.pi)/25  # x方向角频率

    #若t<0,视为尚未开始跟踪轨迹
    def pose(self, t):#目标位置
        if self.FLAG == 1 and t >= 0:
            x = self.R * np.sin(self.w*t)
            y = self.R * (1- np.cos(self.w*t))
            z = self.h_default
            return np.array([[x], [y], [z]])
        elif self.FLAG == 2 and t >= 0:
            x = self.R * np.sin(2*self.w*t)
            y = 2*self.R * (1 - np.cos(self.w*t))
            z = self.h_default
            return np.array([[x], [y], [z]])
        elif self.FLAG == 3 and t >= 0:
            y =  0.5 * 0.6 * t * t
            return np.array([[0.0], [0.0], [self.h_default]])
        elif self.FLAG == 4 and t >= 0:
            # 小圆轨迹：从原点开始，半径2.5m，20秒一圈
            x = self.R4 * np.sin(self.w4*t)
            y = self.R4 * (1 - np.cos(self.w4*t))
            z = self.h_default
            return np.array([[x], [y], [z]])
        elif self.FLAG == 5 and t >= 0:
            # 直线运动：从(-4, 0, 5)匀速飞到(4, 0, 5)，时间为16s
            if t <= self.T5:
                x = self.x_start + self.vx5 * t
            else:
                x = self.x_end
            y = self.y5
            z = self.z5
            return np.array([[x], [y], [z]])
        elif self.FLAG == 6 and t >= 0:
            # 李萨如曲线：x = 2.5sin(wt), y = 2.5sin(2wt), T=25s
            x = self.A6 * np.sin(self.w6 * t)
            y = self.A6 * np.sin(2 * self.w6 * t)
            z = self.h_default
            return np.array([[x], [y], [z]])
        elif self.FLAG == 7 and t >= 0:
            # z轴运动：加速度 az = 0.1*t
            # 速度 vz = 0.05*t²
            # 位置 z = h_default + (1/60)*t³
            x = 0.0
            y = 0.0
            z = self.h_default + (1.0/60.0) * t * t * t
            return np.array([[x], [y], [z]])
        elif t < 0:
            return np.array([[0.0], [0.0], [self.h_default]])
        return None

    def velo(self, t):#目标速度
        if self.FLAG == 1 and t >= 0:
            vx = self.R * self.w * np.cos(self.w*t)
            vy = self.R * self.w * np.sin(self.w*t)
            vz = 0.0
            return np.array([[vx], [vy], [vz]])
        elif self.FLAG == 2 and t >= 0:
            vx = self.R * 2*self.w * np.cos(2*self.w*t)
            vy = 2*self.R * self.w * np.sin(self.w*t)
            vz = 0.0
            return np.array([[vx], [vy], [vz]])
        elif self.FLAG == 3 and t >= 0:
            vy = 0.6 * t
            return np.array([[0.0], [0.0], [0.0]])
        elif self.FLAG == 4 and t >= 0:
            # 小圆轨迹速度
            vx = self.R4 * self.w4 * np.cos(self.w4*t)
            vy = self.R4 * self.w4 * np.sin(self.w4*t)
            vz = 0.0
            return np.array([[vx], [vy], [vz]])
        elif self.FLAG == 5 and t >= 0:
            # 直线运动匀速，速度恒定
            vx = self.vx5
            vy = 0.0
            vz = 0.0
            return np.array([[vx], [vy], [vz]])
        elif self.FLAG == 6 and t >= 0:
            # 李萨如曲线速度
            vx = self.A6 * self.w6 * np.cos(self.w6 * t)
            vy = self.A6 * 2 * self.w6 * np.cos(2 * self.w6 * t)
            vz = 0.0
            return np.array([[vx], [vy], [vz]])
        elif self.FLAG == 7 and t >= 0:
            # z轴运动速度：vz = 0.05*t²
            vx = 0.0
            vy = 0.0
            vz = 0.05 * t * t
            return np.array([[vx], [vy], [vz]])
        elif t < 0:
            return np.array([[0.0], [0.0], [0.0]])
        return None

    def acce(self, t):#目标加速度
        if self.FLAG == 1 and t >= 0:
            ax = -self.R * self.w * self.w * np.sin(self.w*t)
            ay = self.R * self.w * self.w * np.cos(self.w*t)
            az = 0.0
            return np.array([[ax], [ay], [az]])
        elif self.FLAG == 2 and t >= 0:
            ax = -self.R * 2*self.w * 2*self.w * np.sin(2*self.w*t)
            ay = 2*self.R * self.w * self.w * np.cos(self.w*t)
            az = 0.0
            return np.array([[ax], [ay], [az]])
        elif self.FLAG == 3 and t >= 0:
            return np.array([[0.6], [0.6], [0.0]])
        elif self.FLAG == 4 and t >= 0:
            # 小圆轨迹加速度
            ax = -self.R4 * self.w4 * self.w4 * np.sin(self.w4*t)
            ay = self.R4 * self.w4 * self.w4 * np.cos(self.w4*t)
            az = 0.0
            return np.array([[ax], [ay], [az]])
        elif self.FLAG == 5 and t >= 0:
            # 直线运动匀速，加速度为零
            ax = 0.0
            ay = 0.0
            az = 0.0
            return np.array([[ax], [ay], [az]])
        elif self.FLAG == 6 and t >= 0:
            # 李萨如曲线加速度
            ax = -self.A6 * self.w6 * self.w6 * np.sin(self.w6 * t)
            ay = -self.A6 * 2 * self.w6 * 2 * self.w6 * np.sin(2 * self.w6 * t)
            az = 0.0
            return np.array([[ax], [ay], [az]])
        elif self.FLAG == 7 and t >= 0:
            # z轴运动加速度：az = 0.1*t
            ax = 0.0
            ay = 0.0
            az = 0.1 * t
            return np.array([[ax], [ay], [az]])
        elif t < 0:
            return np.array([[0.0], [0.0], [0.0]])
        return None

    def yaw (self, t):#给定偏航角
        if self.FLAG == 1 and t >= 0:
            yaw = 0.0
            return yaw
        elif self.FLAG == 2 and t >= 0:
            yaw = 0.0
            return yaw
        elif self.FLAG == 3 and t >= 0:
            yaw = 0.0
            return yaw
        elif self.FLAG == 4 and t >= 0:
            yaw = 0.0
            return yaw
        elif self.FLAG == 5 and t >= 0:
            yaw = 0.0
            return yaw
        elif self.FLAG == 6 and t >= 0:
            yaw = 0.0
            return yaw
        elif self.FLAG == 7 and t >= 0:
            yaw = 0.0
            return yaw
        elif t < 0:
            return 0.0
        return None

    def jerk(self, t):#目标加速度的导数（三阶导数）
        if self.FLAG == 1 and t >= 0:
            jx = -self.R * self.w * self.w * self.w * np.cos(self.w*t)
            jy = -self.R * self.w * self.w * self.w * np.sin(self.w*t)
            jz = 0.0
            return np.array([[jx], [jy], [jz]])
        elif self.FLAG == 2 and t >= 0:
            jx = -self.R * 2*self.w * 2*self.w * 2*self.w * np.cos(2*self.w*t)
            jy = -2*self.R * self.w * self.w * self.w * np.sin(self.w*t)
            jz = 0.0
            return np.array([[jx], [jy], [jz]])
        elif self.FLAG == 3 and t >= 0:
            return np.array([[0.0], [0.0], [0.0]])
        elif self.FLAG == 4 and t >= 0:
            # 小圆轨迹加速度导数
            jx = -self.R4 * self.w4 * self.w4 * self.w4 * np.cos(self.w4*t)
            jy = -self.R4 * self.w4 * self.w4 * self.w4 * np.sin(self.w4*t)
            jz = 0.0
            return np.array([[jx], [jy], [jz]])
        elif self.FLAG == 5 and t >= 0:
            # 直线运动匀速，三阶导数为零
            jx = 0.0
            jy = 0.0
            jz = 0.0
            return np.array([[jx], [jy], [jz]])
        elif self.FLAG == 6 and t >= 0:
            # 李萨如曲线加速度导数（jerk）
            jx = -self.A6 * self.w6 * self.w6 * self.w6 * np.cos(self.w6 * t)
            jy = -self.A6 * 2 * self.w6 * 2 * self.w6 * 2 * self.w6 * np.cos(2 * self.w6 * t)
            jz = 0.0
            return np.array([[jx], [jy], [jz]])
        elif self.FLAG == 7 and t >= 0:
            # z轴运动加加速度（jerk）：jz = 0.1
            jx = 0.0
            jy = 0.0
            jz = 0.1
            return np.array([[jx], [jy], [jz]])
        elif t < 0:
            return np.array([[0.0], [0.0], [0.0]])
        return None