import math
import numpy as np
import random

def calculate_smoke_valid_duration(drone_speed, theta_deg, t_drop, t_burst, fy1_pos_init, step=0.01):
    # 常量定义
    fy1_point=fy1_pos_init
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

    real_point_1 = (7, 200, 0)
    real_point_2 = (-7, 200, 10)

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

    return smoke_valid_duration, (new_x_drone, new_y_drone, new_z_drone)

# 遗传算法实现

class Population:
    """
    遗传算法种群类。
    针对4个优化变量 (drone_speed, theta, t_drop, t_burst) 进行优化。
    """
    def __init__(self, size, chrom_size, cp, mp, gen_max):
        # 种群信息
        self.individuals = []          # 个体集合, 每个个体有4个染色体
        self.fitness = []              # 个体适应度集
        self.selector_probability = [] # 个体选择概率集合
        self.new_individuals = []      # 新一代个体集合

        # 精英个体信息（迄今为止找到的最佳解）
        self.elitist = {'chromosome': [0, 0, 0, 0], 'fitness': 0.0, 'age': 0}

        self.size = size                   # 种群规模
        self.chromosome_size = chrom_size  # 单个染色体长度（比特数）
        self.crossover_probability = cp    # 交叉概率
        self.mutation_probability = mp     # 变异概率
         
        self.generation_max = gen_max      # 最大进化世代数
        self.age = 0                       # 种群当前代数
        
        # 定义每个优化变量的取值范围
        self.variable_ranges = {
            'drone_speed': [70.0, 140.0],
            'theta': [0.0, 360.0],
            't_drop': [0.0, 20.0],
            't_burst': [0.0, 20.0]
        }

        # 初始化种群
        # 每个个体由4个染色体组成，分别对应4个优化变量
        max_chrom_value = 2 ** self.chromosome_size - 1
        for _ in range(self.size):
            # 每个个体包含4个随机生成的染色体
            self.individuals.append([random.randint(0, max_chrom_value) for _ in range(4)])
            self.new_individuals.append([0, 0, 0, 0])
            self.fitness.append(0)
            self.selector_probability.append(0)

    def decode(self, interval, chromosome):
        """
        将一个染色体 chromosome 映射到给定的 interval 区间内数值
        """
        d = interval[1] - interval[0]
        max_chrom_value = float(2 ** self.chromosome_size - 1)
        return interval[0] + chromosome * d / max_chrom_value
     
    def fitness_func(self, individual_chromosomes):
        """
        适应度函数，可以根据个体的四个染色体计算出该个体的适应度
        """
        drone_speed = self.decode(self.variable_ranges['drone_speed'], individual_chromosomes[0])
        theta = self.decode(self.variable_ranges['theta'], individual_chromosomes[1])
        t_drop = self.decode(self.variable_ranges['t_drop'], individual_chromosomes[2])
        t_burst = self.decode(self.variable_ranges['t_burst'], individual_chromosomes[3])

        duration, _ = calculate_smoke_valid_duration(
            drone_speed=drone_speed,
            theta_deg=theta,
            t_drop=t_drop,
            t_burst=t_burst,
            fy1_pos_init=(17800, 0, 1800)
        )
        return duration
         
    def evaluate(self):
        """评估种群中所有个体的适应度，并计算选择概率"""
        for i in range(self.size):
            self.fitness[i] = self.fitness_func(self.individuals[i])
        
        ft_sum = sum(self.fitness)
        
        # 防止总适应度为0导致除零错误
        if ft_sum == 0:
            # 如果所有个体适应度都为0，则给予均等选择机会
            for i in range(self.size):
                self.selector_probability[i] = 1.0 / self.size
        else:
            for i in range(self.size):
                self.selector_probability[i] = self.fitness[i] / ft_sum
        
        # 计算累积概率，用于轮盘赌选择
        for i in range(1, self.size):
            self.selector_probability[i] += self.selector_probability[i-1]

    def select(self):
        """轮盘赌选择"""
        rand_prob = random.random()
        for i, p in enumerate(self.selector_probability):
            if p > rand_prob:
                return i
        return self.size - 1 # 保证返回一个有效索引

    def cross(self, chrom1, chrom2):
        """交叉"""
        if chrom1 != chrom2 and random.random() < self.crossover_probability:
            # 随机选择交叉点
            cross_point = random.randint(1, self.chromosome_size - 1)
            
            mask = (2 ** self.chromosome_size - 1) << cross_point
            
            # 交换交叉点后的基因
            part1_c1 = chrom1 & mask
            part2_c1 = chrom1 & ~mask
            
            part1_c2 = chrom2 & mask
            part2_c2 = chrom2 & ~mask
            
            new_chrom1 = part1_c1 | part2_c2
            new_chrom2 = part1_c2 | part2_c1
            
            return new_chrom1, new_chrom2
        return chrom1, chrom2

    def mutate(self, chrom):
        """变异"""
        if random.random() < self.mutation_probability:
            # 随机选择一个位进行翻转
            mutation_point = random.randint(1, self.chromosome_size)
            mask = 1 << (mutation_point - 1)
            chrom ^= mask  # 使用异或操作进行位翻转
        return chrom

    def reproduct_elitist(self):
        """找到当前代最佳个体，如果它比历史最佳还好，则更新"""
        current_best_idx = self.fitness.index(max(self.fitness))
        if self.elitist['fitness'] < self.fitness[current_best_idx]:
            self.elitist['fitness'] = self.fitness[current_best_idx]
            # 使用切片进行深拷贝，防止后续被修改
            self.elitist['chromosome'] = self.individuals[current_best_idx][:]
            self.elitist['age'] = self.age

    def evolve(self):
        """执行一代进化：选择、交叉、变异、产生下一代种群"""
        # 计算适应度及选择概率
        self.evaluate()
        
        # 进化操作
        self.reproduct_elitist()

        # 进化操作
        i = 0
        while i < self.size:
            # 选择两个个体，进行交叉与变异，产生新的种群
            parent1_idx = self.select()
            parent2_idx = self.select()
            
            parent1 = self.individuals[parent1_idx]
            parent2 = self.individuals[parent2_idx]
            
            child1 = [0] * 4
            child2 = [0] * 4

            # 对父代的每一对染色体进行交叉和变异
            for j in range(4):
                # 交叉
                (c1, c2) = self.cross(parent1[j], parent2[j])
                # 变异
                child1[j] = self.mutate(c1)
                child2[j] = self.mutate(c2)
            
            self.new_individuals[i] = child1
            if i + 1 < self.size:
                self.new_individuals[i+1] = child2
            
            i += 2
        
        # 更新换代
        self.individuals = self.new_individuals[:]
        
        # 将精英个体替换掉新种群中的最差个体，确保精英不丢失
        # 找到新种群的最差个体
        new_fitness = [self.fitness_func(ind) for ind in self.individuals]
        worst_idx = new_fitness.index(min(new_fitness))
        # 替换
        self.individuals[worst_idx] = self.elitist['chromosome'][:]


    def run(self):
        """运行整个遗传算法进化过程。"""
        for i in range(self.generation_max):
            self.age = i
            self.evolve()
            print(f"世代 {i:3d}: 最佳时长 = {max(self.fitness):.2f}, "
                  f"平均时长 = {sum(self.fitness)/self.size:.2f}, "
                  f"历史最佳 = {self.elitist['fitness']:.2f}")
        
        # 进化结束后，打印最终找到的最佳解
        print("                 优化结束                 ")
        
        best_chromosomes = self.elitist['chromosome']
        best_speed = self.decode(self.variable_ranges['drone_speed'], best_chromosomes[0])
        best_theta = self.decode(self.variable_ranges['theta'], best_chromosomes[1])
        best_tdrop = self.decode(self.variable_ranges['t_drop'], best_chromosomes[2])
        best_tburst = self.decode(self.variable_ranges['t_burst'], best_chromosomes[3])
        
        print(f"  - drone_speed: {best_speed:.4f} m/s")
        print(f"  - theta:       {best_theta:.4f} degrees")
        print(f"  - t_drop:      {best_tdrop:.4f} s")
        print(f"  - t_burst:     {best_tburst:.4f} s")
        print(f"\n最佳遮蔽时长: {self.elitist['fitness']:.2f}")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

SEED_VALUE = 1234
set_all_seeds(SEED_VALUE)

if __name__ == '__main__':
    # --- 遗传算法参数配置 ---
    POPULATION_SIZE = 50       # 种群大小
    CHROMOSOME_BITS = 20       # 单个染色体的位数（精度）
    CROSSOVER_PROB = 0.8       # 交叉概率
    MUTATION_PROB = 0.05       # 变异概率
    MAX_GENERATIONS = 80       # 最大进化代数

    print("开始遗传优化算法...")
    pop = Population(
        size=POPULATION_SIZE, 
        chrom_size=CHROMOSOME_BITS, 
        cp=CROSSOVER_PROB, 
        mp=MUTATION_PROB, 
        gen_max=MAX_GENERATIONS
    )
    pop.run()