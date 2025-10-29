import math
import numpy as np
import random
from sko.GA import GA
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 类定义 (保持不变)
# ==============================================================================

class Missile:
    """一个代表导弹的类。"""
    def __init__(self, missile_id: str, initial_position: tuple, target_position: tuple, speed: int = 300):
        self.id = missile_id
        self.initial_position = tuple(initial_position)
        self.speed = speed
        self.target_position = tuple(target_position)
        self.t_spans = []
        direction_vector = tuple(self.target_position[i] - self.initial_position[i] for i in range(3))
        distance_to_target = math.sqrt(sum(v**2 for v in direction_vector))
        if distance_to_target == 0:
            unit_direction_vector = (0, 0, 0)
        else:
            unit_direction_vector = tuple(v / distance_to_target for v in direction_vector)
        self.vx, self.vy, self.vz = (self.speed * v for v in unit_direction_vector)

    def get_position_at_time(self, t: float) -> tuple:
        delta_x, delta_y, delta_z = self.vx * t, self.vy * t, self.vz * t
        return (self.initial_position[0] + delta_x, self.initial_position[1] + delta_y, self.initial_position[2] + delta_z)
    
    def add_t_span(self, t_begin: float, t_end: float):
        self.t_spans.append(sorted([t_begin, t_end]))

    def get_total_t_span_length(self) -> float:
        if not self.t_spans: return 0
        intervals = sorted(self.t_spans, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if merged[-1][1] < start: merged.append([start, end])
            else: merged[-1][1] = max(merged[-1][1], end)
        return sum(end - start for start, end in merged)

    def __repr__(self):
        return f"Missile(id='{self.id}', pos={self.initial_position}, target={self.target_position})"


class Drone:
    """一个代表无人机的类，作为状态容器。"""
    def __init__(self, drone_id: str, initial_position: tuple):
        self.id = drone_id
        self.initial_position = tuple(initial_position)
        self.associated_missile_id = None
        self.g, self.smoke_center_vz = 9.8, -3.0

    def create_smoke_trajectory_calculator(self, speed, theta_deg, t_drop, t_burst):
        temp_vx, temp_vy = speed * math.cos(math.radians(theta_deg)), speed * math.sin(math.radians(theta_deg))
        release_pos_x, release_pos_y, release_pos_z = self.initial_position[0] + temp_vx * t_drop, self.initial_position[1] + temp_vy * t_drop, self.initial_position[2]
        burst_pos_x, burst_pos_y = release_pos_x + temp_vx * t_burst, release_pos_y + temp_vy * t_burst
        burst_pos_z = release_pos_z + 0.5 * (-self.g) * t_burst**2
        burst_position = (burst_pos_x, burst_pos_y, burst_pos_z)
        smoke_start_time = t_drop + t_burst
        def calculate_smoke_center_position(total_time_t):
            if total_time_t < smoke_start_time: return burst_position
            else:
                smoke_motion_duration = total_time_t - smoke_start_time
                sc_z = burst_position[2] + self.smoke_center_vz * smoke_motion_duration
                return (burst_position[0], burst_position[1], sc_z)
        return calculate_smoke_center_position
    
    def __repr__(self):
        return f"Drone(id='{self.id}', pos={self.initial_position}, assigned_to='{self.associated_missile_id}')"

# ==============================================================================
# 2. 独立的辅助函数 (保持不变)
# ==============================================================================

def point_to_segment_distance(segment_start: tuple, segment_end: tuple, point: tuple) -> float:
    line_vec = tuple(segment_end[i] - segment_start[i] for i in range(3))
    point_vec = tuple(point[i] - segment_start[i] for i in range(3))
    line_len_sq = sum(v**2 for v in line_vec)
    if line_len_sq == 0.0: return math.sqrt(sum(v**2 for v in point_vec))
    dot = sum(point_vec[i] * line_vec[i] for i in range(3))
    t = max(0, min(1, dot / line_len_sq))
    closest_point = tuple(segment_start[i] + t * line_vec[i] for i in range(3))
    dist_sq = sum((point[i] - closest_point[i])**2 for i in range(3))
    return math.sqrt(dist_sq)

# ==============================================================================
# 3. 核心仿真与分配函数 (保持不变)
# ==============================================================================

def calculate_smoke_interception_interval(missile: Missile, drone: Drone, smoke_mission_params: dict, real_point_1: tuple, real_point_2: tuple, threshold: float, smoke_valid_time: float, step: float = 0.01) -> tuple:
    speed, theta_deg, t_drop, t_burst = (smoke_mission_params[k] for k in ['speed', 'theta_deg', 't_drop', 't_burst'])
    smoke_calculator = drone.create_smoke_trajectory_calculator(speed, theta_deg, t_drop, t_burst)
    smoke_start_time = t_drop + t_burst
    simulation_end_time = smoke_start_time + smoke_valid_time
    t_valid_begin, t_valid_end, is_in_valid_interval = 0, 0, False
    for t in np.arange(smoke_start_time, simulation_end_time, step):
        smoke_center_pos, missile_pos = smoke_calculator(t), missile.get_position_at_time(t)
        dist_1, dist_2 = point_to_segment_distance(real_point_1, missile_pos, smoke_center_pos), point_to_segment_distance(real_point_2, missile_pos, smoke_center_pos)
        is_obstructed = (dist_1 < threshold) and (dist_2 < threshold)
        if is_obstructed and not is_in_valid_interval:
            t_valid_begin, is_in_valid_interval = t, True
        elif not is_obstructed and is_in_valid_interval:
            t_valid_end = t
            break
    if is_in_valid_interval and t_valid_end == 0: t_valid_end = simulation_end_time
    return (t_valid_begin, t_valid_end)

def assign_drones_to_missiles(drones: dict, missiles: dict):
    print("\n--- 正在为无人机分配目标... ---")
    for drone_id, drone in drones.items():
        min_distance, closest_missile_id = float('inf'), None
        for missile_id, missile in missiles.items():
            distance = point_to_segment_distance(missile.initial_position, missile.target_position, drone.initial_position)
            if distance < min_distance:
                min_distance, closest_missile_id = distance, missile.id
        drone.associated_missile_id = closest_missile_id
        print(f"无人机 {drone.id} 分配给导弹 {closest_missile_id} (距离: {min_distance:.2f} 米)")

# ==============================================================================
# 4. 遗传算法模块 (保持不变)
# ==============================================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def create_objective_function(drone: Drone, missile: Missile, sim_params: dict):
    def objective_func(p):
        drone_speed, theta, t_drop, t_burst = p
        mission_params = {'speed': drone_speed, 'theta_deg': theta, 't_drop': t_drop, 't_burst': t_burst}
        t_span = calculate_smoke_interception_interval(missile=missile, drone=drone, smoke_mission_params=mission_params, **sim_params)
        duration = t_span[1] - t_span[0]
        return -duration
    return objective_func

# ==============================================================================
# 5. 主程序入口 (绘图部分已修改)
# ==============================================================================

if __name__ == "__main__":
    # --- 1. 场景数据定义与对象实例化 (保持不变) ---
    missile_target = (0, 200, 0)
    missile_positions = {'M1': (20000, 0, 2000), 'M2': (19000, 600, 2100), 'M3': (18000, -600, 1900)}
    drone_positions = {'FY1': (17800, 0, 1800), 'FY2': (12000, 1400, 1400), 'FY3': (6000, -3000, 700), 'FY4': (11000, 2000, 1800), 'FY5': (13000, -2000, 1300)}
    
    missiles = {name: Missile(name, pos, missile_target) for name, pos in missile_positions.items()}
    drones = {name: Drone(name, pos) for name, pos in drone_positions.items()}

    print("--- 对象创建完成 (初始状态) ---")
    for d in drones.values(): print(d)

    # --- 2. 执行分配逻辑 (保持不变) ---
    assign_drones_to_missiles(drones, missiles)
    
    print("\n--- 分配完成后的最终状态 ---")
    for d in drones.values(): print(d)
    
    # --- 3. 使用遗传算法为FY1寻找最优拦截策略 (保持不变) ---
    print("\n" + "="*50)
    print("--- 开始为无人机 FY1 执行遗传算法优化 ---")
    print("="*50)

    SEED_VALUE = 1500
    set_all_seeds(SEED_VALUE)
    
    target_drone_for_ga = drones['FY1']
    target_missile_for_ga = missiles[target_drone_for_ga.associated_missile_id]
    print(f"优化目标: 无人机 '{target_drone_for_ga.id}' vs 导弹 '{target_missile_for_ga.id}'")

    fixed_simulation_params = {
        'real_point_1': (7, 200, 0), 'real_point_2': (-7, 200, 10),
        'threshold': 10.0, 'smoke_valid_time': 20.0
    }

    objective_function = create_objective_function(
        drone=target_drone_for_ga, missile=target_missile_for_ga, sim_params=fixed_simulation_params
    )
    
    ga = GA(func=objective_function, n_dim=4, size_pop=50, max_iter=75, prob_mut=0.01, 
            lb=[70, 0, 0, 0], ub=[140, 180, 5, 5], precision=1e-7)
    
    best_x, best_y = ga.run()
    
    print("\n--- 遗传算法优化结果 ---")
    formatted_x = [round(float(x), 3) for x in best_x]
    max_duration = round(float(-best_y[0]), 3)
    param_names = ['最佳速度 (m/s)', '最佳角度 (deg)', '最佳投放时间 (s)', '最佳爆开时间 (s)']
    for name, val in zip(param_names, formatted_x):
        print(f'{name}: {val}')
    print('最大有效遮蔽时长 (s):', max_duration)
    
    # --- g. 【已修改】绘制优化历史曲线，并解决中文显示问题 ---
    
    # ******** 新增代码开始 ********
    # 设置matplotlib支持中文的字体
    # SimHei 是黑体, Microsoft YaHei 是微软雅黑
    # 请根据你的操作系统选择一个已安装的字体
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False 
    # ******** 新增代码结束 ********

    formatted_y_hist = [-float(y[0]) for y in ga.all_history_Y]
    plt.figure(figsize=(10, 6))
    plt.plot(formatted_y_hist)
    plt.xlabel('迭代次数 (Iteration)')
    plt.ylabel('当前最优值 (最大遮蔽时长)')
    plt.title('遗传算法优化历史 (GA Optimization History for FY1)')
    plt.grid(True)
    plt.show()
