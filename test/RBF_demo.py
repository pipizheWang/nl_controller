import numpy as np
import matplotlib.pyplot as plt

# macOS 下启用中文字体显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RBFNetwork:
    def __init__(self, num_rbf_nodes=5, input_dim=2):
        """RBF网络实现"""
        self.num_rbf_nodes = num_rbf_nodes
        self.input_dim = input_dim

        # 使用文章中的参数配置
        self.b = 7.2

        self.centers = np.array([
            [-2.0, -1.0],
            [-1.0, 0.5],
            [0.0, 0.0],
            [1.0, -0.5],
            [2.0, 1.0]
        ])

        # 文章中随机初始化权重
        self.weights = np.random.normal(0, 0.1, self.num_rbf_nodes)

        # 自适应律参数
        self.gamma = 0.01
        self.weight_clip = 200

    def gaussian_rbf(self, x, center, b):
        """高斯径向基函数"""
        distance_squared = np.sum((x - center) ** 2)
        return np.exp(-distance_squared / (b ** 2))

    def compute_rbf_output(self, x):
        """计算RBF层输出（最简单形式）"""
        h = np.zeros(self.num_rbf_nodes)
        for i in range(self.num_rbf_nodes):
            h[i] = self.gaussian_rbf(x, self.centers[i], self.b)
        return h

    def forward(self, x):
        """前向传播"""
        h = self.compute_rbf_output(x)
        output = np.dot(self.weights, h)
        return output, h

    def update_weights(self, e1, rbf_outputs, dt):
        """带泄露项的最简单自适应律"""
        self.weights += self.gamma * e1 * rbf_outputs * dt
        np.clip(self.weights, -self.weight_clip, self.weight_clip, out=self.weights)

class ControlSystem:
    def __init__(self):
        """控制系统初始化"""
        # PD控制器参数
        self.kp = 25.0  # 比例增益
        self.kd = 10.0  # 微分增益
        
        # RBF网络
        self.rbf_net = RBFNetwork()
        
        # 系统状态记录
        self.reset_system()
        
    def reset_system(self):
        """重置系统状态"""
        self.time_history = []
        self.state_history_pd = []
        self.state_history_rbf = []
        self.control_history_pd = []
        self.control_history_rbf = []
        self.rbf_comp_history = []
        self.error_history_pd = []
        self.error_history_rbf = []
        self.disturbance_history = []
        self.reference_history = []
        
    def generate_disturbance(self, t):
        """生成随机扰动"""
        # 组合多种扰动类型（用于离线生成）
        random_noise = 0.0  # 在线积分时不再引入随机项
        periodic_disturbance = 0.5 * np.sin(3 * t) * np.cos(0.5 * t)
        step_disturbance = 1.0 if 15 < t < 20 else 0.0
        return random_noise + periodic_disturbance + step_disturbance

    def get_disturbance_at(self, t):
        """按时间插值获得扰动（与积分时间轴对齐）"""
        if len(self.time_history) == 0 or len(self.disturbance_history) == 0:
            return 0.0
        return float(np.interp(t, self.time_history, self.disturbance_history))
    
    def reference_signal(self, t):
        """参考信号 - 正弦曲线"""
        amplitude = 2.0
        frequency = 0.5
        return amplitude * np.sin(frequency * t)
    
    def reference_derivative(self, t):
        """参考信号的导数"""
        amplitude = 2.0
        frequency = 0.5
        return amplitude * frequency * np.cos(frequency * t)
    
    def pd_controller(self, x1, x2, x1_ref, x2_ref):
        """普通PD控制器"""
        e1 = x1_ref - x1  # 位置误差
        e2 = x2_ref - x2  # 速度误差
        
        u_pd = self.kp * e1 + self.kd * e2
        return u_pd, e1, e2
    
    def pd_rbf_controller(self, x1, x2, x1_ref, x2_ref, dt):
        """PD + RBF控制器"""
        e1 = x1_ref - x1  # 位置误差
        e2 = x2_ref - x2  # 速度误差

        # PD控制部分
        u_pd = self.kp * e1 + self.kd * e2

        # RBF补偿部分基于跟踪误差
        error_vector = np.array([e1, e2])
        u_rbf, rbf_outputs = self.rbf_net.forward(error_vector)

        # 在线自适应更新
        self.rbf_net.update_weights(e1, rbf_outputs, dt)

        # 控制合成并加简单饱和保护
        u_total = u_pd + u_rbf
        u_total = float(np.clip(u_total, -60.0, 60.0))
        return u_total, e1, e2, u_rbf
    
    def simulate(self, t_span, initial_state):
        """仿真运行"""
        self.reset_system()

        print("开始仿真...")
        print("正在运行普通PD控制器和PD+RBF控制器...")

        # 先设置时间轴，并预生成参考与扰动（固定随机种子确保可复现）
        self.time_history = t_span
        self.reference_history = [self.reference_signal(t) for t in t_span]
        rng = np.random.default_rng(42)
        random_noise_series = 0.3 * rng.normal(0.0, 1.0, size=len(t_span))
        periodic_series = 0.5 * np.sin(3 * t_span) * np.cos(0.5 * t_span)
        step_series = np.where((t_span > 15) & (t_span < 20), 1.0, 0.0)
        self.disturbance_history = list(random_noise_series + periodic_series + step_series)
        dt = float(t_span[1] - t_span[0])

        num_steps = len(t_span)
        states_pd = np.zeros((num_steps, 2))
        states_rbf = np.zeros((num_steps, 2))
        states_pd[0] = initial_state
        states_rbf[0] = initial_state

        # 控制器历史数据
        self.control_history_pd = []
        self.control_history_rbf = []
        self.rbf_comp_history = []
        self.error_history_pd = []
        self.error_history_rbf = []

        # RBF权重初始化
        self.rbf_net.weights = np.zeros_like(self.rbf_net.weights)

        for idx in range(num_steps - 1):
            t = t_span[idx]
            x1_ref = self.reference_history[idx]
            x2_ref = self.reference_derivative(t)
            disturbance = self.disturbance_history[idx]

            # --- 普通PD控制 ---
            x1_pd, x2_pd = states_pd[idx]
            u_pd, e1_pd, e2_pd = self.pd_controller(x1_pd, x2_pd, x1_ref, x2_ref)
            self.control_history_pd.append(u_pd)
            self.error_history_pd.append([e1_pd, e2_pd])

            dx1_pd = x2_pd
            dx2_pd = u_pd + disturbance
            states_pd[idx + 1, 0] = x1_pd + dx1_pd * dt
            states_pd[idx + 1, 1] = x2_pd + dx2_pd * dt

            # --- PD + RBF控制 ---
            x1_rbf, x2_rbf = states_rbf[idx]
            u_total, e1_rbf, e2_rbf, u_rbf = self.pd_rbf_controller(
                x1_rbf, x2_rbf, x1_ref, x2_ref, dt
            )
            self.control_history_rbf.append(u_total)
            self.rbf_comp_history.append(u_rbf)
            self.error_history_rbf.append([e1_rbf, e2_rbf])

            dx1_rbf = x2_rbf
            dx2_rbf = u_total + disturbance
            states_rbf[idx + 1, 0] = x1_rbf + dx1_rbf * dt
            states_rbf[idx + 1, 1] = x2_rbf + dx2_rbf * dt

        # 使控制历史长度与时间轴一致（最后一个控制输入重复上一时刻）
        if self.control_history_pd:
            self.control_history_pd.append(self.control_history_pd[-1])
        if self.control_history_rbf:
            self.control_history_rbf.append(self.control_history_rbf[-1])
            self.rbf_comp_history.append(self.rbf_comp_history[-1])

        self.state_history_pd = states_pd
        self.state_history_rbf = states_rbf

        print("仿真完成!")
    
    def plot_results(self):
        """绘制仿真结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PD控制器 vs PD+RBF控制器性能对比', fontsize=16)
        
        t = self.time_history
        
        # 位置跟踪对比
        axes[0, 0].plot(t, self.reference_history, 'k--', linewidth=2, label='参考信号')
        axes[0, 0].plot(t, self.state_history_pd[:, 0], 'b-', linewidth=1.5, label='普通PD')
        axes[0, 0].plot(t, self.state_history_rbf[:, 0], 'r-', linewidth=1.5, label='PD+RBF')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('位置 x1')
        axes[0, 0].set_title('位置跟踪性能')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 跟踪误差对比
        error_pd = np.array(self.reference_history) - self.state_history_pd[:, 0]
        error_rbf = np.array(self.reference_history) - self.state_history_rbf[:, 0]
        
        axes[0, 1].plot(t, error_pd, 'b-', linewidth=1.5, label='普通PD误差')
        axes[0, 1].plot(t, error_rbf, 'r-', linewidth=1.5, label='PD+RBF误差')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('跟踪误差')
        axes[0, 1].set_title('跟踪误差对比')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 控制输入对比
        n_pd = min(len(t) - 1, len(self.control_history_pd))
        if n_pd > 0:
            axes[1, 0].plot(t[1:1 + n_pd], self.control_history_pd[:n_pd], 'b-', linewidth=1.5, label='普通PD')
        n_rbf = min(len(t) - 1, len(self.control_history_rbf))
        if n_rbf > 0:
            axes[1, 0].plot(t[1:1 + n_rbf], self.control_history_rbf[:n_rbf], 'r-', linewidth=1.5, label='PD+RBF')
        if self.rbf_comp_history:
            n_comp = min(len(t) - 1, len(self.rbf_comp_history))
            axes[1, 0].plot(t[1:1 + n_comp], self.rbf_comp_history[:n_comp], 'r--', linewidth=1.2, label='RBF补偿')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('控制输入 u')
        axes[1, 0].set_title('控制输入对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 扰动信号
        axes[1, 1].plot(t[1:], self.disturbance_history[1:], 'g-', linewidth=1.5, label='扰动信号')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('扰动 d')
        axes[1, 1].set_title('系统扰动')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 性能指标计算
        rmse_pd = np.sqrt(np.mean(error_pd**2))
        rmse_rbf = np.sqrt(np.mean(error_rbf**2))
        max_error_pd = np.max(np.abs(error_pd))
        max_error_rbf = np.max(np.abs(error_rbf))
        
        print("\n性能指标对比:")
        print(f"{'指标':<15} {'普通PD':<12} {'PD+RBF':<12} {'改善率':<10}")
        print("-" * 55)
        print(f"{'RMSE':<15} {rmse_pd:<12.4f} {rmse_rbf:<12.4f} {((rmse_pd-rmse_rbf)/rmse_pd*100):<10.2f}%")
        print(f"{'最大误差':<15} {max_error_pd:<12.4f} {max_error_rbf:<12.4f} {((max_error_pd-max_error_rbf)/max_error_pd*100):<10.2f}%")

def main():
    """主函数"""
    # 创建控制系统
    control_sys = ControlSystem()
    
    # 仿真参数设置
    t_start = 0
    t_end = 30
    dt = 0.01
    t_span = np.arange(t_start, t_end, dt)
    
    # 初始状态 [x1, x2]
    initial_state = [0.0, 0.0]
    
    print("=== 控制系统仿真开始 ===")
    print(f"仿真时间: {t_start}s ~ {t_end}s")
    print(f"采样时间: {dt}s")
    print(f"初始状态: x1={initial_state[0]}, x2={initial_state[1]}")
    print(f"PD参数: Kp={control_sys.kp}, Kd={control_sys.kd}")
    print(f"RBF参数: 节点数={control_sys.rbf_net.num_rbf_nodes}, 宽度={control_sys.rbf_net.b}")
    
    # 运行仿真
    control_sys.simulate(t_span, initial_state)
    
    # 绘制结果
    control_sys.plot_results()

if __name__ == "__main__":
    main()
