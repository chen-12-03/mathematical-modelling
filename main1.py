import math
import numpy as np
# --- FY1飞行部分---
fy1_point = (17800, 0, 1800)
fake_point = (0, 0, 0)
drone_speed = 120
drone_move_time = 1.5

# 计算运动方向向量
vx = drone_speed * -1
vy = drone_speed * 0
vz = drone_speed * 0

# 计算1.5秒后的位置
delta_x = vx * drone_move_time
delta_y = vy * drone_move_time
delta_z = vz * drone_move_time

new_x_drone = fy1_point[0] + delta_x
new_y_drone = fy1_point[1] + delta_y
new_z_drone = fy1_point[2] + delta_z

# print(f"\n运动 {drone_move_time} 秒后，drone的位置为: ({new_x_drone:.2f}, {new_y_drone:.2f}, {new_z_drone:.2f})")

# --- 烟雾弹下落部分 ---
g = 9.8
smoke_bomb_fall_time = 3.6

# 计算smoke_bomb下落3.6秒后的速度
smoke_bomb_initial_vx = vx
smoke_bomb_initial_vy = vy
smoke_bomb_initial_vz = vz

smoke_bomb_vx_final = smoke_bomb_initial_vx
smoke_bomb_vy_final = smoke_bomb_initial_vy

delta_vz_gravity = -g * smoke_bomb_fall_time

# smoke_bomb在z方向的最终速度
smoke_bomb_vz_final = smoke_bomb_initial_vz + delta_vz_gravity

# 计算smoke_bomb下落3.6秒后的位置
smoke_bomb_initial_x = new_x_drone
smoke_bomb_initial_y = new_y_drone
smoke_bomb_initial_z = new_z_drone

delta_x_smoke_bomb = smoke_bomb_initial_vx * smoke_bomb_fall_time
delta_y_smoke_bomb = smoke_bomb_initial_vy * smoke_bomb_fall_time
delta_z_smoke_bomb = smoke_bomb_initial_vz * smoke_bomb_fall_time + 0.5 * (-g) * smoke_bomb_fall_time**2

# smoke_bomb在z方向的最终位置
smoke_bomb_final_x = smoke_bomb_initial_x + delta_x_smoke_bomb
smoke_bomb_final_y = smoke_bomb_initial_y + delta_y_smoke_bomb
smoke_bomb_final_z = smoke_bomb_initial_z + delta_z_smoke_bomb

# print(f"\n运动 {drone_move_time+smoke_bomb_fall_time} 秒后的位置:")
# print(f"  x: {smoke_bomb_final_x:.2f}")
# print(f"  y: {smoke_bomb_final_y:.2f}")
# print(f"  z: {smoke_bomb_final_z:.2f}")

# --- smoke_center运动部分 ---
smoke_center_vx = 0
smoke_center_vy = 0
smoke_center_vz = -3

smoke_center_start_time_total = drone_move_time + smoke_bomb_fall_time

def calculate_smoke_center_position(total_time_t):
    """
    计算smoke_center在总时间t后的坐标。
    返回值:smoke_center在总时间t后的(x, y, z)坐标。
    """
    if total_time_t < smoke_center_start_time_total:
        # smoke_center还没有开始运动
        # print(f"在总时间 {total_time_t:.2f}s, smoke_center尚未开始运动。")
        return (smoke_bomb_final_x, smoke_bomb_final_y, smoke_bomb_final_z)
    else:
        # smoke_center已经在运动
        smoke_center_motion_duration = total_time_t - smoke_center_start_time_total

        sc_x = smoke_bomb_final_x + smoke_center_vx * smoke_center_motion_duration
        sc_y = smoke_bomb_final_y + smoke_center_vy * smoke_center_motion_duration
        sc_z = smoke_bomb_final_z + smoke_center_vz * smoke_center_motion_duration

        return (sc_x, sc_y, sc_z)

# --- M1运动部分 ---
M1_initial_point = (20000, 0, 2000)
M1_speed = 300

# 计算 M1 运动方向向量
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

# 分解 M1 的速度
M1_vx = M1_speed * M1_unit_direction_vector[0]
M1_vy = M1_speed * M1_unit_direction_vector[1]
M1_vz = M1_speed * M1_unit_direction_vector[2]

def calculate_M1_position(total_time_t):
    """
    计算M1在总时间t后的坐标。
    返回值:M1在总时间t后的(x, y, z)坐标。
    """
    M1_delta_x = M1_vx * total_time_t
    M1_delta_y = M1_vy * total_time_t
    M1_delta_z = M1_vz * total_time_t

    # M1 在t时刻的坐标
    M1_final_x = M1_initial_point[0] + M1_delta_x
    M1_final_y = M1_initial_point[1] + M1_delta_y
    M1_final_z = M1_initial_point[2] + M1_delta_z

    return (M1_final_x, M1_final_y, M1_final_z)

def point_to_segment_distance(segment_start, segment_end, point):
    """
    计算一个点到一条线段的最短路径长度。
    参数:
        segment_start: 线段起始点坐标 (x, y, z)。
        segment_end: 线段结束点坐标 (x, y, z)。
        point: 要计算距离的点坐标 (x, y, z)。
    返回值:点到线段的最短路径长度。
    """
    def distance_between_points(p1, p2):
        """两点间欧式距离"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    def dot_product(v1, v2):
        """向量点积"""
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    def vector_subtract(p1, p2):
        """向量差"""
        return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    AB = vector_subtract(segment_end, segment_start)
    AP = vector_subtract(point, segment_start)
    
    # 计算向量 AB 的模平方
    AB_squared = dot_product(AB, AB)

    if AB_squared == 0:
        return distance_between_points(point, segment_start)

    # t 表示点 P 在 AB 直线上的投影位置，相对于 A 点 (t=0) 和 B 点 (t=1)
    t = dot_product(AP, AB) / AB_squared

    # 如果投影点在 A 点之前
    if t < 0:
        return distance_between_points(point, segment_start)
    # 如果投影点在 B 点之后
    elif t > 1:
        return distance_between_points(point, segment_end)
    # 如果投影点在线段 AB 上
    else:
        # 计算投影点 Q 的坐标
        # Q = A + t * AB
        projection_point = (
            segment_start[0] + t * AB[0],
            segment_start[1] + t * AB[1],
            segment_start[2] + t * AB[2]
        )
        # 返回点 P 到投影点 Q 的距离
        return distance_between_points(point, projection_point)
    
real_point_1 = (7, 200, 0)
real_point_2 = (-7, 200, 10)
smoke_valid_time = 20 # 烟雾运动20秒后消散

# 统计 dist_1 和 dist_2 均小于 10 的时间长度（步长0.01秒，单位为秒）
smoke_valid_duration = 0
step = 0.001
for t in np.arange(0, smoke_center_start_time_total + smoke_valid_time + 0.01, step):
    smoke_center_pos = calculate_smoke_center_position(t)
    M1_pos = calculate_M1_position(t)
    dist_1 = point_to_segment_distance(real_point_1, M1_pos, smoke_center_pos)
    dist_2 = point_to_segment_distance(real_point_2, M1_pos, smoke_center_pos)
    if dist_1 < 10 and dist_2 < 10:
        smoke_valid_duration += step

print(f"\n有效遮挡时间为：{smoke_valid_duration:.3f} 秒")
