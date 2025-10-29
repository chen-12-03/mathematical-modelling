import math
import numpy as np
import random
import itertools

def calculate_smoke_valid_duration(drone_speed, theta_deg, t_drop, t_burst, fyx_pos_init, step=0.01, t_bias=0):
    # 常量定义
    fyx_point=fyx_pos_init
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

    new_x_drone = fyx_point[0] + delta_x
    new_y_drone = fyx_point[1] + delta_y
    new_z_drone = fyx_point[2] + delta_z

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
        t_m1 = total_time_t + t_bias
        M1_delta_x = M1_vx * t_m1
        M1_delta_y = M1_vy * t_m1
        M1_delta_z = M1_vz * t_m1
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

    real_point_1 = (7, 200, 0)
    real_point_2 = (-7, 200, 10)

    start_loop_time = smoke_center_start_time_total
    end_loop_time = smoke_center_start_time_total + smoke_valid_time + step

    t_valid_begin = 0
    t_valid_end = 0
    for t in np.arange(start_loop_time, end_loop_time, step):
        smoke_center_pos = calculate_smoke_center_position(t)
        M1_pos = calculate_M1_position(t)
        dist_1 = point_to_segment_distance(real_point_1, M1_pos, smoke_center_pos)
        dist_2 = point_to_segment_distance(real_point_2, M1_pos, smoke_center_pos)
        if dist_1 < 10 and dist_2 < 10 and t_valid_begin == 0:
            t_valid_begin = t
        if dist_1 >=10 and dist_2 >= 10 and t_valid_begin != 0:
            t_valid_end = t
            break
    #      有效遮挡时间区间                 投放点坐标                              起爆点坐标
    return (t_valid_begin, t_valid_end), (new_x_drone, new_y_drone, new_z_drone), (smoke_bomb_final_x, smoke_bomb_final_y, smoke_bomb_final_z)

def calculate_smoke_valid_duration_3(drone_speed_1, theta_deg_1, t_drop_1, t_burst_1,
                                     drone_speed_2, theta_deg_2, t_drop_2, t_burst_2,
                                     drone_speed_3, theta_deg_3, t_drop_3, t_burst_3, 
                                     fyx_pos_init_1, fyx_pos_init_2, fyx_pos_init_3, step=0.01):
    t_duration_1, drop_pos_1, burst_pos_1 = calculate_smoke_valid_duration(drone_speed_1, theta_deg_1, t_drop_1, t_burst_1, fyx_pos_init_1, step=step)
    t_duration_2, drop_pos_2, burst_pos_2 = calculate_smoke_valid_duration(drone_speed_2, theta_deg_2, t_drop_2, t_burst_2, fyx_pos_init_2, step=step)
    t_duration_3, drop_pos_3, burst_pos_3 = calculate_smoke_valid_duration(drone_speed_3, theta_deg_3, t_drop_3, t_burst_3, fyx_pos_init_3, step=step)

    t_span_1 = [t_duration_1[0], t_duration_1[1]]
    t_span_2 = [t_duration_2[0], t_duration_2[1]]
    t_span_3 = [t_duration_3[0], t_duration_3[1]]
    def union_interval_length(intervals):
        # intervals: [(start1, end1), (start2, end2), ...]
        # 先按起点排序
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(list(interval))
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        # 计算总长度
        total = sum(end - start for start, end in merged)
        return total
    
    intervals = [t_span_1, t_span_2, t_span_3]
    total_duration = union_interval_length(intervals)

    t_1 = t_span_1[1] - t_span_1[0]
    t_2 = t_span_2[1] - t_span_2[0]
    t_3 = t_span_3[1] - t_span_3[0]
    t_spans = (t_1, t_2, t_3)
    smoke_inf_1 = (drop_pos_1, burst_pos_1, t_1)
    smoke_inf_2 = (drop_pos_2, burst_pos_2, t_2)
    smoke_inf_3 = (drop_pos_3, burst_pos_3, t_3)

    return  total_duration, smoke_inf_1, smoke_inf_2, smoke_inf_3, t_spans