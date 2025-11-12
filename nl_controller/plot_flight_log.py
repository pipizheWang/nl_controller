#!/usr/bin/env python3
"""
飞行日志绘图工具
读取CSV格式的飞行日志并绘制位置响应、跟踪误差和轨迹曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import sys


class FlightLogPlotter:
    def __init__(self, log_file_path):
        """
        初始化飞行日志绘图器
        
        Args:
            log_file_path: 日志文件路径
        """
        self.log_file_path = Path(log_file_path)
        self.data = None
        
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
            'vx': np.array([float(row['vx']) for row in rows]),
            'vy': np.array([float(row['vy']) for row in rows]),
            'vz': np.array([float(row['vz']) for row in rows]),
            'roll': np.array([float(row['roll']) for row in rows]),
            'pitch': np.array([float(row['pitch']) for row in rows]),
            'yaw': np.array([float(row['yaw']) for row in rows]),
            'x_des': np.array([float(row['x_des']) for row in rows]),
            'y_des': np.array([float(row['y_des']) for row in rows]),
            'z_des': np.array([float(row['z_des']) for row in rows]),
        }
        
        # 将时间戳转换为相对时间（从0开始）
        self.data['time'] = self.data['timestamp'] - self.data['timestamp'][0]
        
        # 计算跟踪误差
        self.data['error_x'] = self.data['x'] - self.data['x_des']
        self.data['error_y'] = self.data['y'] - self.data['y_des']
        self.data['error_z'] = self.data['z'] - self.data['z_des']
        
        print(f"成功加载 {len(rows)} 条数据记录")
        print(f"飞行时长: {self.data['time'][-1]:.2f} 秒")
        
    def plot_position_response(self):
        """绘制三轴位置和期望位置的响应曲线（3列1行）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Position Response', fontsize=14, fontweight='bold')
        
        axes_labels = ['X Axis', 'Y Axis', 'Z Axis']
        pos_keys = ['x', 'y', 'z']
        des_keys = ['x_des', 'y_des', 'z_des']
        
        for i, (ax, label, pos_key, des_key) in enumerate(zip(axes, axes_labels, pos_keys, des_keys)):
            ax.plot(self.data['time'], self.data[pos_key], 'b-', linewidth=2, label='Actual')
            ax.plot(self.data['time'], self.data[des_key], 'r--', linewidth=2, label='Desired')
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Position (m)', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_tracking_error(self):
        """绘制三轴轨迹跟踪误差曲线（3列1行）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Tracking Error', fontsize=14, fontweight='bold')
        
        axes_labels = ['X Error', 'Y Error', 'Z Error']
        error_keys = ['error_x', 'error_y', 'error_z']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (ax, label, error_key, color) in enumerate(zip(axes, axes_labels, error_keys, colors)):
            ax.plot(self.data['time'], self.data[error_key], color=color, linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Error (m)', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 显示统计信息
            mean_error = np.mean(np.abs(self.data[error_key]))
            max_error = np.max(np.abs(self.data[error_key]))
            ax.text(0.02, 0.98, f'Mean: {mean_error:.3f}m\nMax: {max_error:.3f}m',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_attitude(self):
        """绘制三个姿态角的曲线（3列1行）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Attitude Angles', fontsize=14, fontweight='bold')
        
        axes_labels = ['Roll', 'Pitch', 'Yaw']
        attitude_keys = ['roll', 'pitch', 'yaw']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (ax, label, att_key, color) in enumerate(zip(axes, axes_labels, attitude_keys, colors)):
            # 数据已经是度数，无需转换
            angle_deg = self.data[att_key]
            ax.plot(self.data['time'], angle_deg, color=color, linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Angle (deg)', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 显示统计信息
            mean_angle = np.mean(np.abs(angle_deg))
            max_angle = np.max(np.abs(angle_deg))
            ax.text(0.02, 0.98, f'Mean: {mean_angle:.2f}°\nMax: {max_angle:.2f}°',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_trajectory(self):
        """绘制无人机轨迹曲线（3D和XY平面，2列1行）"""
        fig = plt.figure(figsize=(14, 6))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 绘制实际轨迹
        ax1.plot(self.data['x'], self.data['y'], self.data['z'], 
                'b-', linewidth=2, label='Actual', alpha=0.8)
        
        # 绘制期望轨迹
        ax1.plot(self.data['x_des'], self.data['y_des'], self.data['z_des'], 
                'r--', linewidth=2, label='Desired', alpha=0.8)
        
        # 标记起点和终点
        ax1.scatter(self.data['x'][0], self.data['y'][0], self.data['z'][0], 
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(self.data['x'][-1], self.data['y'][-1], self.data['z'][-1], 
                   c='red', s=100, marker='s', label='End', zorder=5)
        
        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_zlabel('Z (m)', fontsize=11)
        ax1.set_title('3D Trajectory', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 设置相等的坐标轴比例
        max_range = np.array([
            self.data['x'].max() - self.data['x'].min(),
            self.data['y'].max() - self.data['y'].min(),
            self.data['z'].max() - self.data['z'].min()
        ]).max() / 2.0
        
        mid_x = (self.data['x'].max() + self.data['x'].min()) * 0.5
        mid_y = (self.data['y'].max() + self.data['y'].min()) * 0.5
        mid_z = (self.data['z'].max() + self.data['z'].min()) * 0.5
        
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # XY平面轨迹图
        ax2 = fig.add_subplot(122)
        
        # 绘制实际轨迹
        ax2.plot(self.data['x'], self.data['y'], 'b-', linewidth=2, 
                label='Actual', alpha=0.8)
        
        # 绘制期望轨迹
        ax2.plot(self.data['x_des'], self.data['y_des'], 'r--', linewidth=2, 
                label='Desired', alpha=0.8)
        
        # 标记起点和终点
        ax2.scatter(self.data['x'][0], self.data['y'][0], 
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax2.scatter(self.data['x'][-1], self.data['y'][-1], 
                   c='red', s=100, marker='s', label='End', zorder=5)
        
        ax2.set_xlabel('X (m)', fontsize=11)
        ax2.set_ylabel('Y (m)', fontsize=11)
        ax2.set_title('XY Plane Trajectory', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_all(self, save_dir=None):
        """
        绘制所有图表
        
        Args:
            save_dir: 保存图片的目录，如果为None则只显示不保存
        """
        if self.data is None:
            self.load_data()
        
        # 使用英文标签，避免中文字体问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("\nGenerating plots...")
        
        # 1. 位置响应曲线
        print("1. Plotting position response...")
        fig1 = self.plot_position_response()
        
        # 2. 跟踪误差曲线
        print("2. Plotting tracking error...")
        fig2 = self.plot_tracking_error()
        
        # 3. 姿态角曲线
        print("3. Plotting attitude angles...")
        fig3 = self.plot_attitude()
        
        # 4. 轨迹曲线
        print("4. Plotting trajectories...")
        fig4 = self.plot_trajectory()
        
        # 保存图片
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            log_name = self.log_file_path.stem  # 获取不带扩展名的文件名
            
            fig1_path = save_path / f'{log_name}_position_response.png'
            fig2_path = save_path / f'{log_name}_tracking_error.png'
            fig3_path = save_path / f'{log_name}_attitude.png'
            fig4_path = save_path / f'{log_name}_trajectory.png'
            
            print(f"\nSaving plots to: {save_path}")
            fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
            print(f"  - {fig1_path.name}")
            fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
            print(f"  - {fig2_path.name}")
            fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
            print(f"  - {fig3_path.name}")
            fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
            print(f"  - {fig4_path.name}")
        
        print("\nAll plots generated successfully!")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Flight Log Plotting Tool')
    parser.add_argument('log_file', type=str, nargs='?', 
                       help='Log file path (CSV format)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots (default: log/plots)')
    parser.add_argument('--list', action='store_true',
                       help='List all log files in log directory')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show plot windows')
    
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
        # 如果没有指定文件，使用最新的日志文件（按修改时间排序）
        if log_dir.exists():
            log_files = list(log_dir.glob('*.csv'))
            if log_files:
                # 按修改时间排序，最新的在最后
                log_files.sort(key=lambda f: f.stat().st_mtime)
                log_file_path = log_files[-1]
                print(f"No log file specified, using the latest: {log_file_path.name}")
            else:
                print(f"Error: No log files found in {log_dir}")
                print("Use --list to see available log files")
                sys.exit(1)
        else:
            print(f"Error: Log directory does not exist: {log_dir}")
            sys.exit(1)
    
    try:
        # 创建绘图器并绘制
        plotter = FlightLogPlotter(log_file_path)
        plotter.load_data()
        
        # 如果没有指定保存目录，默认保存到log目录下的plots子目录
        if args.save_dir is None:
            save_dir = log_dir / 'plots'
        else:
            save_dir = args.save_dir
        
        # 设置不显示图形窗口
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        
        plotter.plot_all(save_dir=save_dir)
        
        # 显示图形窗口（仅在非no-show模式）
        if not args.no_show:
            plt.show()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
