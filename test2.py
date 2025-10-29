import math
class Missile:
    """一个精简的导弹类，用于存储ID和轨迹的起止点。"""
    def __init__(self, missile_id: str, initial_position: tuple, target_position: tuple):
        self.id = missile_id
        self.initial_position = tuple(initial_position)
        self.target_position = tuple(target_position)

    def __repr__(self):
        return f"Missile(id='{self.id}', pos={self.initial_position}, target={self.target_position})"


class Drone:
    """一个精简的无人机类，用于存储ID、位置，并接收分配的导弹ID。"""
    def __init__(self, drone_id: str, initial_position: tuple):
        self.id = drone_id
        self.initial_position = tuple(initial_position)
        self.associated_missile_id = None  # 等待被分配

    def __repr__(self):
        return f"Drone(id='{self.id}', pos={self.initial_position}, assigned_to='{self.associated_missile_id}')"

def point_to_segment_distance(segment_start: tuple, segment_end: tuple, point: tuple) -> float:
    """计算3D空间中一个点到一个线段的最短距离。"""
    line_vec = tuple(segment_end[i] - segment_start[i] for i in range(3))
    point_vec = tuple(point[i] - segment_start[i] for i in range(3))
    
    line_len_sq = sum(v**2 for v in line_vec)
    if line_len_sq == 0.0:
        return math.sqrt(sum(v**2 for v in point_vec))
        
    dot = sum(point_vec[i] * line_vec[i] for i in range(3))
    t = max(0, min(1, dot / line_len_sq))
    
    closest_point = tuple(segment_start[i] + t * line_vec[i] for i in range(3))
    dist_sq = sum((point[i] - closest_point[i])**2 for i in range(3))
    
    return math.sqrt(dist_sq)

def assign_drones_to_missiles(drones: dict, missiles: dict):
    """遍历所有无人机，根据到导弹轨迹的最近距离为其分配目标。"""
    print("\n--- 正在为无人机分配目标... ---")
    for drone_id, drone in drones.items():
        min_distance = float('inf')
        closest_missile_id = None
        
        for missile_id, missile in missiles.items():
            distance = point_to_segment_distance(
                missile.initial_position, 
                missile.target_position, 
                drone.initial_position
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_missile_id = missile.id
        
        drone.associated_missile_id = closest_missile_id
        print(f"无人机 {drone.id} 分配给导弹 {closest_missile_id} (距离: {min_distance:.2f} 米)")

if __name__ == "__main__":
    # --- 1. 定义场景数据 ---
    missile_target = (0, 200, 0)
    missile_positions = {
        'M1': (20000, 0, 2000), 
        'M2': (19000, 600, 2100), 
        'M3': (18000, -600, 1900)
    }
    drone_positions = {
        'FY1': (17800, 0, 1800), 
        'FY2': (12000, 1400, 1400), 
        'FY3': (6000, -3000, 700), 
        'FY4': (11000, 2000, 1800), 
        'FY5': (13000, -2000, 1300)
    }
    
    # --- 2. 实例化对象 ---
    missiles = {name: Missile(name, pos, missile_target) for name, pos in missile_positions.items()}
    drones = {name: Drone(name, pos) for name, pos in drone_positions.items()}

    print("--- 对象创建完成 (初始状态) ---")
    for d in drones.values():
        print(d)

    # --- 3. 执行核心的分配逻辑 ---
    assign_drones_to_missiles(drones, missiles)
    
    print("\n--- 分配完成后的最终状态 ---")
    for d in drones.values():
        print(d)