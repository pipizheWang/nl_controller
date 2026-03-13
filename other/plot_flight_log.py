#!/usr/bin/env python3
"""
飞行日志绘图工具
读取CSV格式的飞行日志并绘制位置响应、跟踪误差和轨迹曲线
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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

    @staticmethod
    def _find_first_available_font(candidates):
        """返回候选字体列表中首个可用字体名，不存在则返回None。"""
        available = {font.name for font in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                return name
        return None

    def configure_plot_fonts(self):
        """配置绘图字体：中文优先宋体，英文字母/数字优先Times New Roman。"""
        latin_candidates = [
            'Times New Roman',
            'Times',
            'Nimbus Roman',
            'Liberation Serif',
            'DejaVu Serif',
        ]
        songti_candidates = [
            'SimSun',
            'Songti SC',
            'STSong',
            'AR PL UMing CN',
            'Source Han Serif SC',
            'Noto Serif CJK SC',
        ]

        latin_font = self._find_first_available_font(latin_candidates)
        chinese_font = self._find_first_available_font(songti_candidates)

        if latin_font is None:
            latin_font = 'DejaVu Serif'
            print('警告: 未检测到 Times New Roman，将使用 DejaVu Serif 作为英文字体。')

        if chinese_font is None:
            raise RuntimeError(
                '未检测到可用宋体中文字体。请在Linux安装宋体或宋体风格字体后重试，'
                '例如安装 AR PL UMing CN / Noto Serif CJK SC。'
            )

        # 按字符回退：英文字母/数字优先使用Times，中文回退到宋体。
        plt.rcParams['font.family'] = [latin_font, chinese_font]
        plt.rcParams['font.serif'] = [latin_font, chinese_font]
        plt.rcParams['axes.unicode_minus'] = False

        # 数学文本中的字母和数字也使用英文字体，避免与主图不一致。
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = latin_font
        plt.rcParams['mathtext.it'] = latin_font + ':italic'
        plt.rcParams['mathtext.bf'] = latin_font + ':bold'

        print(f'字体配置完成: 英文/数字={latin_font}, 中文={chinese_font}')
        
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
        
        # 尝试加载自适应参数（如果存在）
        self.has_adaptive_params = False
        if 'a_x_hat' in rows[0] and 'a_y_hat' in rows[0] and 'rho_x_hat' in rows[0] and 'rho_y_hat' in rows[0]:
            self.data['a_x_hat'] = np.array([float(row['a_x_hat']) for row in rows])
            self.data['a_y_hat'] = np.array([float(row['a_y_hat']) for row in rows])
            self.data['rho_x_hat'] = np.array([float(row['rho_x_hat']) for row in rows])
            self.data['rho_y_hat'] = np.array([float(row['rho_y_hat']) for row in rows])
            self.has_adaptive_params = True
            print(f"检测到自适应参数数据")
        
        # 尝试加载姿态指令（如果存在）
        self.has_attitude_commands = False
        if 'roll_cmd' in rows[0] and 'pitch_cmd' in rows[0]:
            self.data['roll_cmd'] = np.array([float(row['roll_cmd']) for row in rows])
            self.data['pitch_cmd'] = np.array([float(row['pitch_cmd']) for row in rows])
            self.has_attitude_commands = True
            print(f"检测到姿态指令数据")
        
        # 将时间戳转换为相对时间（从0开始）
        self.data['time'] = self.data['timestamp'] - self.data['timestamp'][0]
        
        # 计算跟踪误差
        self.data['error_x'] = self.data['x'] - self.data['x_des']
        self.data['error_y'] = self.data['y'] - self.data['y_des']
        self.data['error_z'] = self.data['z'] - self.data['z_des']
        
        # 计算欧式距离跟踪误差
        self.data['tracking_error'] = np.sqrt(
            self.data['error_x']**2 + 
            self.data['error_y']**2 + 
            self.data['error_z']**2
        )
        
        print(f"成功加载 {len(rows)} 条数据记录")
        print(f"飞行时长: {self.data['time'][-1]:.2f} 秒")
        print(f"跟踪误差统计 - 平均: {np.mean(self.data['tracking_error']):.4f}m, 最大: {np.max(self.data['tracking_error']):.4f}m")
        
    def plot_position_response(self):
        """绘制三轴位置和期望位置的响应曲线（3列1行）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('位置响应曲线', fontsize=14, fontweight='bold')
        
        axes_labels = ['X轴', 'Y轴', 'Z轴']
        pos_keys = ['x', 'y', 'z']
        des_keys = ['x_des', 'y_des', 'z_des']
        
        for i, (ax, label, pos_key, des_key) in enumerate(zip(axes, axes_labels, pos_keys, des_keys)):
            ax.plot(self.data['time'], self.data[pos_key], 'b-', linewidth=2, label='实际值')
            ax.plot(self.data['time'], self.data[des_key], 'r--', linewidth=2, label='期望值')
            ax.set_xlabel('时间（秒）', fontsize=11)
            ax.set_ylabel('位置（米）', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_tracking_error(self):
        """绘制三轴轨迹跟踪误差曲线（3列1行）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('跟踪误差曲线', fontsize=14, fontweight='bold')
        
        axes_labels = ['X轴误差', 'Y轴误差', 'Z轴误差']
        error_keys = ['error_x', 'error_y', 'error_z']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (ax, label, error_key, color) in enumerate(zip(axes, axes_labels, error_keys, colors)):
            ax.plot(self.data['time'], self.data[error_key], color=color, linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('时间（秒）', fontsize=11)
            ax.set_ylabel('误差（米）', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 显示统计信息
            mean_error = np.mean(np.abs(self.data[error_key]))
            max_error = np.max(np.abs(self.data[error_key]))
            ax.text(
                0.02,
                0.98,
                f'平均值: {mean_error:.3f}米\n最大值: {max_error:.3f}米',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9,
            )
        
        plt.tight_layout()
        return fig
    
    def plot_attitude(self):
        """绘制三个姿态角的曲线（3列1行）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('姿态角曲线', fontsize=14, fontweight='bold')
        
        axes_labels = ['横滚角', '俯仰角', '偏航角']
        attitude_keys = ['roll', 'pitch', 'yaw']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (ax, label, att_key, color) in enumerate(zip(axes, axes_labels, attitude_keys, colors)):
            # 数据已经是度数，无需转换
            angle_deg = self.data[att_key]
            ax.plot(self.data['time'], angle_deg, color=color, linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('时间（秒）', fontsize=11)
            ax.set_ylabel('角度（度）', fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 显示统计信息
            mean_angle = np.mean(np.abs(angle_deg))
            max_angle = np.max(np.abs(angle_deg))
            ax.text(
                0.02,
                0.98,
                f'平均值: {mean_angle:.2f}°\n最大值: {max_angle:.2f}°',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9,
            )
        
        plt.tight_layout()
        return fig
    
    def plot_adaptive_parameters(self):
        """绘制自适应参数曲线（4列1行）"""
        if not self.has_adaptive_params:
            print("警告: 日志中没有自适应参数数据")
            return None
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle('自适应参数曲线', fontsize=14, fontweight='bold')
        
        param_labels = ['a_x_hat', 'a_y_hat', 'rho_x_hat', 'rho_y_hat']
        display_labels = ['a_x估计值', 'a_y估计值', 'rho_x估计值', 'rho_y估计值']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (ax, param_key, display_label, color) in enumerate(zip(axes, param_labels, display_labels, colors)):
            ax.plot(self.data['time'], self.data[param_key], color=color, linewidth=2)
            ax.set_xlabel('时间（秒）', fontsize=11)
            ax.set_ylabel('参数值', fontsize=11)
            ax.set_title(display_label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 显示统计信息
            mean_val = np.mean(self.data[param_key])
            final_val = self.data[param_key][-1]
            ax.text(
                0.02,
                0.98,
                f'平均值: {mean_val:.6f}\n最终值: {final_val:.6f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9,
            )
        
        plt.tight_layout()
        return fig
    
    def plot_attitude_commands(self):
        """绘制姿态角和指令对比曲线（2列1行）"""
        if not self.has_attitude_commands:
            print("警告: 日志中没有姿态指令数据")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle('姿态角与指令对比', fontsize=14, fontweight='bold')
        
        # Roll对比
        axes[0].plot(self.data['time'], self.data['roll'], 'b-', linewidth=2, label='横滚角实际值')
        axes[0].plot(self.data['time'], self.data['roll_cmd'], 'r--', linewidth=2, label='横滚角指令值')
        axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
        axes[0].set_xlabel('时间（秒）', fontsize=11)
        axes[0].set_ylabel('角度（度）', fontsize=11)
        axes[0].set_title('横滚角跟踪', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best', fontsize=9)
        
        # 显示Roll跟踪统计信息
        roll_error = self.data['roll'] - self.data['roll_cmd']
        mean_roll_error = np.mean(np.abs(roll_error))
        max_roll_error = np.max(np.abs(roll_error))
        axes[0].text(0.02, 0.98, f'平均误差: {mean_roll_error:.2f}°\n最大误差: {max_roll_error:.2f}°',
                   transform=axes[0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        
        # Pitch对比
        axes[1].plot(self.data['time'], self.data['pitch'], 'b-', linewidth=2, label='俯仰角实际值')
        axes[1].plot(self.data['time'], self.data['pitch_cmd'], 'r--', linewidth=2, label='俯仰角指令值')
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
        axes[1].set_xlabel('时间（秒）', fontsize=11)
        axes[1].set_ylabel('角度（度）', fontsize=11)
        axes[1].set_title('俯仰角跟踪', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best', fontsize=9)
        
        # 显示Pitch跟踪统计信息
        pitch_error = self.data['pitch'] - self.data['pitch_cmd']
        mean_pitch_error = np.mean(np.abs(pitch_error))
        max_pitch_error = np.max(np.abs(pitch_error))
        axes[1].text(0.02, 0.98, f'平均误差: {mean_pitch_error:.2f}°\n最大误差: {max_pitch_error:.2f}°',
                   transform=axes[1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_trajectory(self):
        """绘制无人机轨迹曲线（3D和XY平面，2列1行），用颜色表示跟踪误差"""
        fig = plt.figure(figsize=(14, 6))
        
        # 获取跟踪误差用于颜色映射
        errors = self.data['tracking_error']
        norm = Normalize(vmin=0, vmax=0.2)  # 固定误差范围0-0.2m，便于对比不同控制方法
        cmap = plt.cm.jet  # 使用jet颜色映射：蓝色表示小误差，红色表示大误差
        
        # 3D轨迹图
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 绘制期望轨迹
        ax1.plot(self.data['x_des'], self.data['y_des'], self.data['z_des'], 
            'k--', linewidth=1.5, label='期望轨迹', alpha=0.6)
        
        # 绘制实际轨迹（用颜色表示误差）
        points = np.array([self.data['x'], self.data['y'], self.data['z']]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 创建3D LineCollection
        lc3d = Line3DCollection(segments, cmap=cmap, norm=norm)
        lc3d.set_array(errors[:-1])
        lc3d.set_linewidth(2)
        ax1.add_collection3d(lc3d)
        
        # 标记起点和终点
        ax1.scatter(self.data['x'][0], self.data['y'][0], self.data['z'][0], 
                   c='green', s=100, marker='o', label='起点', zorder=5)
        ax1.scatter(self.data['x'][-1], self.data['y'][-1], self.data['z'][-1], 
                   c='red', s=100, marker='s', label='终点', zorder=5)
        
        ax1.set_xlabel('X轴（米）', fontsize=11)
        ax1.set_ylabel('Y轴（米）', fontsize=11)
        ax1.set_zlabel('Z轴（米）', fontsize=11)
        ax1.set_title('三维轨迹（颜色表示跟踪误差）', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        all_x = np.concatenate([self.data['x'], self.data['x_des']])
        all_y = np.concatenate([self.data['y'], self.data['y_des']])
        all_z = np.concatenate([self.data['z'], self.data['z_des']])
        
        max_range = np.array([
            all_x.max() - all_x.min(),
            all_y.max() - all_y.min(),
            all_z.max() - all_z.min()
        ]).max() / 2.0
        
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # XY平面轨迹图
        ax2 = fig.add_subplot(122)
        
        # 绘制期望轨迹
        ax2.plot(self.data['x_des'], self.data['y_des'], 'k--', linewidth=1.5, 
            label='期望轨迹', alpha=0.6)
        
        # 绘制实际轨迹（用颜色表示误差）
        points_2d = np.array([self.data['x'], self.data['y']]).T.reshape(-1, 1, 2)
        segments_2d = np.concatenate([points_2d[:-1], points_2d[1:]], axis=1)
        
        lc2d = LineCollection(segments_2d, cmap=cmap, norm=norm)
        lc2d.set_array(errors[:-1])
        lc2d.set_linewidth(2)
        ax2.add_collection(lc2d)
        
        # 标记起点和终点
        ax2.scatter(self.data['x'][0], self.data['y'][0], 
                   c='green', s=100, marker='o', label='起点', zorder=5)
        ax2.scatter(self.data['x'][-1], self.data['y'][-1], 
                   c='red', s=100, marker='s', label='终点', zorder=5)
        
        ax2.set_xlabel('X轴（米）', fontsize=11)
        ax2.set_ylabel('Y轴（米）', fontsize=11)
        ax2.set_title('XY平面轨迹（颜色表示跟踪误差）', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.autoscale_view()
        
        # 添加颜色条
        cbar = fig.colorbar(lc2d, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('跟踪误差（米）', fontsize=10)
        
        # 显示误差统计信息
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        fig.text(0.5, 0.02, f'平均误差: {mean_error:.4f}米 | 最大误差: {max_error:.4f}米', 
                ha='center', fontsize=11, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig
    
    def plot_all(self, save_dir=None):
        """
        绘制所有图表
        
        Args:
            save_dir: 保存图片的目录，如果为None则只显示不保存
        """
        if self.data is None:
            self.load_data()
        
        # 配置字体，保证Linux环境下中文与英文字体正确显示
        self.configure_plot_fonts()
        
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
        
        # 5. 自适应参数曲线（如果存在）
        fig5 = None
        if self.has_adaptive_params:
            print("5. Plotting adaptive parameters...")
            fig5 = self.plot_adaptive_parameters()
        
        # 6. 姿态指令对比（如果存在）
        fig6 = None
        if self.has_attitude_commands:
            print("6. Plotting attitude commands...")
            fig6 = self.plot_attitude_commands()
        
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
            
            if fig5 is not None:
                fig5_path = save_path / f'{log_name}_adaptive_params.png'
                fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
                print(f"  - {fig5_path.name}")
            
            if fig6 is not None:
                fig6_path = save_path / f'{log_name}_attitude_commands.png'
                fig6.savefig(fig6_path, dpi=300, bbox_inches='tight')
                print(f"  - {fig6_path.name}")
        
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
