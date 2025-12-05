#!/usr/bin/env python3
"""
无人机3D轨迹动画工具
读取CSV格式的飞行日志并生成3D动画，展示无人机的飞行轨迹和姿态变化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import csv
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d


class QuadrotorModel:
    """四旋翼无人机模型"""
    
    def __init__(self, arm_length=0.3):
        """
        初始化四旋翼模型
        
        Args:
            arm_length: 机臂长度（米）
        """
        self.arm_length = arm_length
        self.rotor_radius = arm_length * 0.25  # 旋翼半径
        
        # 定义机身框架（X型配置）
        # 四个机臂的端点位置（机体坐标系）
        self.arm_angles = np.array([45, 135, 225, 315]) * np.pi / 180  # 转换为弧度
        
    def get_body_frame(self):
        """
        获取机体框架的顶点（机体坐标系）
        
        Returns:
            机臂端点坐标列表
        """
        # 计算四个机臂端点
        arms = []
        for angle in self.arm_angles:
            x = self.arm_length * np.cos(angle)
            y = self.arm_length * np.sin(angle)
            arms.append([x, y, 0])
        
        return np.array(arms)
    
    def rotation_matrix(self, roll, pitch, yaw):
        """
        计算旋转矩阵（ZYX欧拉角，航空顺序）
        
        Args:
            roll: 滚转角（弧度）
            pitch: 俯仰角（弧度）
            yaw: 偏航角（弧度）
            
        Returns:
            3x3旋转矩阵
        """
        # 滚转矩阵（绕X轴）
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # 俯仰矩阵（绕Y轴）
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # 偏航矩阵（绕Z轴）
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵：R = Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def transform(self, position, roll, pitch, yaw):
        """
        将机体坐标系的点转换到世界坐标系
        
        Args:
            position: 无人机位置 [x, y, z]
            roll: 滚转角（弧度）
            pitch: 俯仰角（弧度）
            yaw: 偏航角（弧度）
            
        Returns:
            转换后的机臂端点坐标和旋翼圆心坐标
        """
        # 获取机体框架
        body_frame = self.get_body_frame()
        
        # 计算旋转矩阵
        R = self.rotation_matrix(roll, pitch, yaw)
        
        # 旋转并平移
        world_frame = (R @ body_frame.T).T + position
        
        return world_frame


class TrajectoryAnimator:
    """轨迹动画生成器"""
    
    def __init__(self, log_file_path, arm_length=0.5):
        """
        初始化轨迹动画器
        
        Args:
            log_file_path: 日志文件路径
            arm_length: 无人机机臂长度
        """
        self.log_file_path = Path(log_file_path)
        self.data = None
        self.quadrotor = QuadrotorModel(arm_length=arm_length)
        self.fig = None
        self.ax = None
        self.frame_skip = 1  # 跳帧设置，用于加速动画
        
    def load_data(self):
        """从CSV文件加载飞行日志数据"""
        if not self.log_file_path.exists():
            raise FileNotFoundError(f"日志文件不存在: {self.log_file_path}")
        
        # 读取CSV文件
        with open(self.log_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if len(rows) == 0:
            raise ValueError("日志文件为空")
        
        # 提取数据
        self.data = {
            'timestamp': np.array([float(row['timestamp']) for row in rows]),
            'x': np.array([float(row['x']) for row in rows]),
            'y': np.array([float(row['y']) for row in rows]),
            'z': np.array([float(row['z']) for row in rows]),
            'roll': np.array([float(row['roll']) for row in rows]),
            'pitch': np.array([float(row['pitch']) for row in rows]),
            'yaw': np.array([float(row['yaw']) for row in rows]),
            'x_des': np.array([float(row['x_des']) for row in rows]),
            'y_des': np.array([float(row['y_des']) for row in rows]),
            'z_des': np.array([float(row['z_des']) for row in rows]),
        }
        
        # 将时间戳转换为相对时间
        self.data['time'] = self.data['timestamp'] - self.data['timestamp'][0]
        
        print(f"成功加载 {len(rows)} 条数据记录")
        print(f"飞行时长: {self.data['time'][-1]:.2f} 秒")
        
        # 自动调整跳帧率，使动画大约有300-500帧
        total_frames = len(rows)
        target_frames = 400
        self.frame_skip = max(1, total_frames // target_frames)
        print(f"跳帧率: {self.frame_skip} (动画将有约 {total_frames // self.frame_skip} 帧)")
        
    def setup_plot(self):
        """设置绘图环境"""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置标题
        self.fig.suptitle('Quadrotor Trajectory Animation', fontsize=16, fontweight='bold')
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
        
        # 计算坐标轴范围
        all_x = np.concatenate([self.data['x'], self.data['x_des']])
        all_y = np.concatenate([self.data['y'], self.data['y_des']])
        all_z = np.concatenate([self.data['z'], self.data['z_des']])
        
        # 找到最大范围，使用相等的坐标轴比例
        max_range = np.array([
            all_x.max() - all_x.min(),
            all_y.max() - all_y.min(),
            all_z.max() - all_z.min()
        ]).max() / 2.0
        
        # 添加一些边距
        max_range *= 1.2
        
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # Z轴固定范围为4-6米
        self.ax.set_zlim(4, 6)
        
        # 设置视角
        self.ax.view_init(elev=25, azim=45)
        
        # 绘制期望轨迹（半透明红色虚线）
        self.ax.plot(self.data['x_des'], self.data['y_des'], self.data['z_des'],
                    'r--', linewidth=2, alpha=0.3, label='Desired Path')
        
        # 绘制起点标记
        self.ax.scatter(self.data['x'][0], self.data['y'][0], self.data['z'][0],
                       c='green', s=200, marker='o', label='Start', zorder=10)
        
        # 绘制终点标记
        self.ax.scatter(self.data['x_des'][-1], self.data['y_des'][-1], self.data['z_des'][-1],
                       c='red', s=200, marker='s', label='Goal', zorder=10)
        
        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
    def init_animation(self):
        """初始化动画元素"""
        # 初始化轨迹线（蓝色）
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2.5, 
                                            alpha=0.8, label='Actual Path')
        
        # 初始化无人机机臂（4条线段，从中心到各个机臂端点）
        self.arm_lines = []
        for i in range(4):
            line, = self.ax.plot([], [], [], 'k-', linewidth=3)
            self.arm_lines.append(line)
        
        # 初始化旋翼圆圈（4个）
        self.rotor_circles = []
        for i in range(4):
            # 在3D中绘制圆圈比较复杂，我们用多边形近似
            line, = self.ax.plot([], [], [], 'gray', linewidth=2, alpha=0.7)
            self.rotor_circles.append(line)
        
        # 添加信息文本
        self.info_text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return [self.trajectory_line] + self.arm_lines + self.rotor_circles + [self.info_text]
    
    def update_frame(self, frame):
        """更新动画帧"""
        # 实际数据索引（考虑跳帧）
        idx = frame * self.frame_skip
        if idx >= len(self.data['x']):
            idx = len(self.data['x']) - 1
        
        # 当前位置和姿态
        x = self.data['x'][idx]
        y = self.data['y'][idx]
        z = self.data['z'][idx]
        roll = self.data['roll'][idx]
        pitch = self.data['pitch'][idx]
        yaw = self.data['yaw'][idx]
        
        # 更新已飞过的轨迹
        self.trajectory_line.set_data(self.data['x'][:idx+1], self.data['y'][:idx+1])
        self.trajectory_line.set_3d_properties(self.data['z'][:idx+1])
        
        # 获取转换后的机臂端点
        position = np.array([x, y, z])
        # CSV中的姿态角是度数，需要转换为弧度传递给transform函数
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        arm_endpoints = self.quadrotor.transform(position, roll_rad, pitch_rad, yaw_rad)
        
        # 更新机臂
        for i, line in enumerate(self.arm_lines):
            # 从中心到机臂端点
            line.set_data([position[0], arm_endpoints[i, 0]], 
                         [position[1], arm_endpoints[i, 1]])
            line.set_3d_properties([position[2], arm_endpoints[i, 2]])
        
        # 更新旋翼（绘制为圆圈）
        for i, line in enumerate(self.rotor_circles):
            # 在每个机臂端点绘制旋翼圆圈
            center = arm_endpoints[i]
            
            # 生成圆圈点（在垂直于机臂的平面上）
            n_points = 20
            theta = np.linspace(0, 2*np.pi, n_points)
            
            # 旋翼在机体XY平面上的圆（简化处理）
            radius = self.quadrotor.rotor_radius
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)
            circle_z = np.zeros_like(theta)
            
            # 将圆圈转换到世界坐标系
            R = self.quadrotor.rotation_matrix(roll_rad, pitch_rad, yaw_rad)
            circle_points = np.vstack([circle_x, circle_y, circle_z])
            world_circle = (R @ circle_points).T + center
            
            line.set_data(world_circle[:, 0], world_circle[:, 1])
            line.set_3d_properties(world_circle[:, 2])
        
        # 更新信息文本
        time_str = f"Time: {self.data['time'][idx]:.2f}s / {self.data['time'][-1]:.2f}s"
        pos_str = f"Position: ({x:.2f}, {y:.2f}, {z:.2f}) m"
        # CSV中的姿态角已经是度数，无需转换
        att_str = f"Attitude: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°"
        
        # 计算误差
        error_x = x - self.data['x_des'][idx]
        error_y = y - self.data['y_des'][idx]
        error_z = z - self.data['z_des'][idx]
        error_total = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        error_str = f"Tracking Error: {error_total:.3f} m"
        
        self.info_text.set_text(f"{time_str}\n{pos_str}\n{att_str}\n{error_str}")
        
        return [self.trajectory_line] + self.arm_lines + self.rotor_circles + [self.info_text]
    
    def create_animation(self, interval=50, save_path=None):
        """
        创建动画
        
        Args:
            interval: 帧间隔（毫秒）
            save_path: 保存路径（如果为None则只显示）
        """
        if self.data is None:
            self.load_data()
        
        self.setup_plot()
        
        # 计算帧数
        n_frames = len(self.data['x']) // self.frame_skip
        
        print(f"\n开始生成动画...")
        print(f"总帧数: {n_frames}")
        print(f"帧间隔: {interval} ms")
        
        # 创建动画
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=n_frames,
            interval=interval,
            blit=False,  # 3D动画不能使用blit
            repeat=True
        )
        
        # 保存或显示
        if save_path is not None:
            save_path = Path(save_path)
            print(f"\n正在保存动画到: {save_path}")
            
            if save_path.suffix == '.gif':
                writer = PillowWriter(fps=1000//interval)
                anim.save(save_path, writer=writer, dpi=100)
                print(f"动画已保存为GIF: {save_path}")
            elif save_path.suffix == '.mp4':
                anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=100)
                print(f"动画已保存为MP4: {save_path}")
            else:
                print(f"警告: 不支持的文件格式 {save_path.suffix}，将保存为GIF")
                save_path = save_path.with_suffix('.gif')
                writer = PillowWriter(fps=1000//interval)
                anim.save(save_path, writer=writer, dpi=100)
                print(f"动画已保存: {save_path}")
        
        return anim


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quadrotor Trajectory Animation Tool')
    parser.add_argument('log_file', type=str, nargs='?',
                       help='Log file path (CSV format)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save animation to file (support .gif and .mp4)')
    parser.add_argument('--interval', type=int, default=50,
                       help='Frame interval in milliseconds (default: 50)')
    parser.add_argument('--arm-length', type=float, default=0.5,
                       help='Quadrotor arm length in meters (default: 0.5)')
    parser.add_argument('--list', action='store_true',
                       help='List all log files in log directory')
    
    args = parser.parse_args()
    
    # 获取log目录路径
    current_file = Path(__file__).resolve()
    package_dir = current_file.parent.parent
    log_dir = package_dir / 'log'
    
    # 列出所有日志文件
    if args.list:
        if log_dir.exists():
            log_files = sorted(log_dir.glob('*.csv'))
            if log_files:
                print(f"\n在 {log_dir} 中找到以下日志文件：")
                for i, log_file in enumerate(log_files, 1):
                    size_kb = log_file.stat().st_size / 1024
                    print(f"  {i}. {log_file.name} ({size_kb:.1f} KB)")
            else:
                print(f"在 {log_dir} 中没有找到日志文件")
        else:
            print(f"日志目录不存在: {log_dir}")
        return
    
    # 确定日志文件路径
    if args.log_file:
        log_file_path = Path(args.log_file)
    else:
        # 如果没有指定文件，使用最新的日志文件
        if log_dir.exists():
            log_files = list(log_dir.glob('*.csv'))
            if log_files:
                log_files.sort(key=lambda f: f.stat().st_mtime)
                log_file_path = log_files[-1]
                print(f"未指定日志文件，使用最新的: {log_file_path.name}")
            else:
                print(f"错误: 在 {log_dir} 中没有找到日志文件")
                print("使用 --list 查看可用的日志文件")
                sys.exit(1)
        else:
            print(f"错误: 日志目录不存在: {log_dir}")
            sys.exit(1)
    
    # 确定保存路径
    save_path = None
    if args.save:
        save_path = Path(args.save)
    else:
        # 默认保存为GIF到log目录
        log_name = log_file_path.stem
        save_path = log_dir / f'{log_name}_animation.gif'
        print(f"未指定保存路径，将保存到: {save_path}")
    
    try:
        # 创建动画器
        animator = TrajectoryAnimator(log_file_path, arm_length=args.arm_length)
        animator.load_data()
        
        # 生成动画
        anim = animator.create_animation(interval=args.interval, save_path=save_path)
        
        # 显示动画窗口
        print("\n显示动画窗口（关闭窗口以退出）...")
        plt.show()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
