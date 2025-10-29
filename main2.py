import math
import numpy as np

def calculate_smoke_valid_duration(drone_speed, theta_deg, t_drop, t_burst, step=0.01):
    # 常量定义
    fy1_point=(17800, 0, 1800)
    fake_point=(0, 0, 0)
    g=9.8
    smoke_center_vz=-3
    smoke_valid_time=20

    # --- FY1飞行部分---
    vx = drone_speed * np.cos(np.radians(theta_deg))
    vy = drone_speed * np.sin(np.radians(theta_deg))
    vz = 0

    delta_x = vx * t_drop
    delta_y = vy * t_drop
    delta_z = vz * t_drop

    new_x_drone = fy1_point[0] + delta_x
    new_y_drone = fy1_point[1] + delta_y
    new_z_drone = fy1_point[2] + delta_z

    # --- 烟雾弹下落部分 ---
    smoke_bomb_initial_vx = vx
    smoke_bomb_initial_vy = vy
    smoke_bomb_initial_vz = vz

    delta_vz_gravity = -g * t_burst

    # smoke_bomb在z方向的最终速度
    smoke_bomb_vz_final = smoke_bomb_initial_vz + delta_vz_gravity

    # 计算smoke_bomb下落smoke_bomb_fall_time秒后的位置
    smoke_bomb_initial_x = new_x_drone
    smoke_bomb_initial_y = new_y_drone
    smoke_bomb_initial_z = new_z_drone

    delta_x_smoke_bomb = smoke_bomb_initial_vx * t_burst
    delta_y_smoke_bomb = smoke_bomb_initial_vy * t_burst
    delta_z_smoke_bomb = smoke_bomb_initial_vz * t_burst + 0.5 * (-g) * t_burst**2

    smoke_bomb_final_x = smoke_bomb_initial_x + delta_x_smoke_bomb
    smoke_bomb_final_y = smoke_bomb_initial_y + delta_y_smoke_bomb
    smoke_bomb_final_z = smoke_bomb_initial_z + delta_z_smoke_bomb

    # --- smoke_center运动部分 ---
    smoke_center_vx = 0
    smoke_center_vy = 0

    smoke_center_start_time_total = t_drop + t_burst

    def calculate_smoke_center_position(total_time_t):
        if total_time_t < smoke_center_start_time_total:
            return (smoke_bomb_final_x, smoke_bomb_final_y, smoke_bomb_final_z)
        else:
            smoke_center_motion_duration = total_time_t - smoke_center_start_time_total
            sc_x = smoke_bomb_final_x + smoke_center_vx * smoke_center_motion_duration
            sc_y = smoke_bomb_final_y + smoke_center_vy * smoke_center_motion_duration
            sc_z = smoke_bomb_final_z + smoke_center_vz * smoke_center_motion_duration
            return (sc_x, sc_y, sc_z)

    # --- M1运动部分 ---
    M1_initial_point = (20000, 0, 2000)
    M1_speed = 300

    M1_direction_vector = (
        fake_point[0] - M1_initial_point[0],
        fake_point[1] - M1_initial_point[1],
        fake_point[2] - M1_initial_point[2]
    )
    M1_distance = math.sqrt(
        M1_direction_vector[0]**2 + M1_direction_vector[1]**2 + M1_direction_vector[2]**2
    )
    M1_unit_direction_vector = (
        M1_direction_vector[0] / M1_distance,
        M1_direction_vector[1] / M1_distance,
        M1_direction_vector[2] / M1_distance
    )

    M1_vx = M1_speed * M1_unit_direction_vector[0]
    M1_vy = M1_speed * M1_unit_direction_vector[1]
    M1_vz = M1_speed * M1_unit_direction_vector[2]

    def calculate_M1_position(total_time_t):
        M1_delta_x = M1_vx * total_time_t
        M1_delta_y = M1_vy * total_time_t
        M1_delta_z = M1_vz * total_time_t
        M1_final_x = M1_initial_point[0] + M1_delta_x
        M1_final_y = M1_initial_point[1] + M1_delta_y
        M1_final_z = M1_initial_point[2] + M1_delta_z
        return (M1_final_x, M1_final_y, M1_final_z)

    def point_to_segment_distance(segment_start, segment_end, point):
        def distance_between_points(p1, p2):
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        def dot_product(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        def vector_subtract(p1, p2):
            return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

        AB = vector_subtract(segment_end, segment_start)
        AP = vector_subtract(point, segment_start)
        AB_squared = dot_product(AB, AB)
        if AB_squared == 0:
            return distance_between_points(point, segment_start)
        t = dot_product(AP, AB) / AB_squared
        if t < 0:
            return distance_between_points(point, segment_start)
        elif t > 1:
            return distance_between_points(point, segment_end)
        else:
            projection_point = (
                segment_start[0] + t * AB[0],
                segment_start[1] + t * AB[1],
                segment_start[2] + t * AB[2]
            )
            return distance_between_points(point, projection_point)

    real_point_1 = (0, 200, 0)
    real_point_2 = (0, 200, 10)

    smoke_valid_duration = 0

    start_loop_time = smoke_center_start_time_total
    end_loop_time = smoke_center_start_time_total + smoke_valid_time + step
    
    smoke_valid_duration = 0
    for t in np.arange(start_loop_time, end_loop_time, step):
        smoke_center_pos = calculate_smoke_center_position(t)
        M1_pos = calculate_M1_position(t)
        dist_1 = point_to_segment_distance(real_point_1, M1_pos, smoke_center_pos)
        dist_2 = point_to_segment_distance(real_point_2, M1_pos, smoke_center_pos)
        if dist_1 < 10 and dist_2 < 10:
            smoke_valid_duration += step


    return smoke_valid_duration

# --- PSO 代码 ---
# 粒子类 (Particle Class)
class Particle:
    def __init__(self, bounds):
        self.bounds = np.array(bounds)
        self.position = np.array([np.random.uniform(low, high) for low, high in self.bounds])
        self.velocity = np.zeros(len(bounds))
        self.best_position = np.copy(self.position)
        self.best_fitness = -np.inf

# 粒子群优化算法
def particle_swarm_optimization(objective_function, bounds, num_particles, max_iterations, initial_guess=None):
    num_dimensions = len(bounds)
    bounds_arr = np.array(bounds)
    
    # 参数设置
    w_max, w_min = 0.9, 0.4
    c1_initial, c1_final = 2.5, 0.5
    c2_initial, c2_final = 0.5, 2.5

    # 初始化粒子群
    particles = [Particle(bounds) for _ in range(num_particles)]
    
    # 初始化全局最佳
    global_best_position = np.zeros(num_dimensions)
    global_best_fitness = -np.inf

    # 提供初始猜测点，计算其适应度并设置为初始全局最优
    if initial_guess is not None:
        initial_guess_pos = np.array(initial_guess)
        initial_fitness = objective_function(*initial_guess_pos)
        
        # 将其设置为初始的全局最优
        global_best_position = initial_guess_pos
        global_best_fitness = initial_fitness
        print(f"设置初始解 Duration 为: {initial_fitness:.3f}")

    # 开始迭代
    for i in range(max_iterations):
        # 动态计算 w, c1, c2
        w = w_max - (w_max - w_min) * i / max_iterations
        c1 = c1_initial - (c1_initial - c1_final) * i / max_iterations
        c2 = c2_initial + (c2_final - c2_initial) * i / max_iterations

        for particle in particles:
            fitness = objective_function(*particle.position)
            
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = np.copy(particle.position)
            
            # 这里的比较会自动处理，如果初始化的 global_best 更好，它会保持
            # 如果某个随机粒子更好，它会被更新
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle.position)
        
        # 更新粒子速度和位置
        for particle in particles:
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            
            cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
            social_velocity = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity
            
            particle.position += particle.velocity
            
            particle.position = np.clip(particle.position, bounds_arr[:, 0], bounds_arr[:, 1])

        if (i + 1) % 10 == 0:
            print(f"迭代次数: {i + 1}/{max_iterations}, 最佳时长: {global_best_fitness:.3f} | "
                  f"参数: w={w:.2f}, c1={c1:.2f}, c2={c2:.2f}")

    return global_best_position, global_best_fitness

if __name__ == '__main__':
    BOUNDS = [
        [70, 140],       # drone_speed (m/s)
        [0, 360],        # theta (角度)
        [0, 10],         # t_drop (s)
        [0, 10]          # t_burst (s)
    ]
    
    NUM_PARTICLES = 100
    MAX_ITERATIONS = 50

    # 定义初始猜测点
    # 顺序必须为: [drone_speed, theta, t_drop, t_burst]
    my_initial_guess = [120, 180, 1.5, 3.6]

    print("开始使用带有初始猜测的粒子群算法进行优化...")

    # 传递猜测点
    best_solution, best_duration = particle_swarm_optimization(
        objective_function=calculate_smoke_valid_duration,
        bounds=BOUNDS,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        initial_guess=my_initial_guess
    )

    print("\n优化完成")
    print("===================================")
    print(f"找到的最大 Duration: {best_duration:.3f}")
    print("对应的最佳参数组合为:")
    print(f"  - drone_speed: {best_solution[0]:.4f} m/s")
    print(f"  - theta: {best_solution[1]:.4f} 度")
    print(f"  - t_drop: {best_solution[2]:.4f} s")
    print(f"  - t_burst: {best_solution[3]:.4f} s")
    print("===================================")