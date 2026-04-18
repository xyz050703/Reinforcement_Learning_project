import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleMineEnv(gym.Env):
    def __init__(self, config_path="./conf/north_pit_mine.json"):
        super(SimpleMineEnv, self).__init__()
        
        # 1. 加载数据
        with open(config_path, "r", encoding="utf-8") as f:
            self.mine_data = json.load(f)
            
        # 获取路网矩阵 (转为 numpy 数组方便查询)
        self.l2d_matrix = np.array(self.mine_data['road']['l2d_road_matrix'])
        self.d2l_matrix = np.array(self.mine_data['road']['d2l_road_matrix'])
        
        self.num_load_sites = len(self.l2d_matrix)    # 5 个装载点
        self.num_dump_sites = len(self.l2d_matrix[0]) # 5 个卸载点
        self.total_trucks = 71
        
        # 2. 定义动作空间 (Action Space)
        # 假设我们只做一个最基础的调度：卡车卸完矿后，决定去哪个装载点 (0 到 4)
        self.action_space = spaces.Discrete(self.num_load_sites)
        
        # 3. 定义状态空间 (Observation Space)
        # 状态我们需要知道：5个装载点的排队车辆数 + 5个卸载点的排队车辆数 = 10个维度的向量
        # 假设单个站点最多排队 71 辆车
        self.observation_space = spaces.Box(
            low=0, high=self.total_trucks, shape=(10,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 初始化矿区状态：所有站点的排队数量清零
        self.load_queues = np.zeros(self.num_load_sites, dtype=np.float32)
        self.dump_queues = np.zeros(self.num_dump_sites, dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 100 # 假设一个回合跑 100 次调度决策
        
        # 组装初始状态
        obs = np.concatenate([self.load_queues, self.dump_queues])
        return obs, {}

    def step(self, action):
        """
        action: 智能体选择的装载点 ID (0-4)
        """
        self.current_step += 1
        
        # --- 这里是你未来要写的“核心物理逻辑” ---
        # 比如：卡车去了装载点 `action`，那边的排队数量就 +1
        self.load_queues[action] += 1 
        
        # 模拟一些随机卸矿完成的车辆，让队列动态变化
        self.load_queues = np.maximum(0, self.load_queues - np.random.randint(0, 2, size=5))
        
        # --- 设计奖励函数 (Reward) ---
        # 核心目标是少排队。排队越多的地方，惩罚越大
        queue_penalty = -np.sum(self.load_queues) 
        reward = queue_penalty
        
        # 判断是否结束
        done = self.current_step >= self.max_steps
        truncated = False
        
        # 组装新状态
        obs = np.concatenate([self.load_queues, self.dump_queues])
        
        return obs, reward, done, truncated, {}

# --- 测试环境是否工作正常 ---
if __name__ == "__main__":
    env = SimpleMineEnv()
    obs, info = env.reset()
    print("初始化状态 (10维排队数):", obs)
    
    # 模拟智能体随便选了 3 个动作 (比如去了 1号、0号、4号装载点)
    for a in [1, 0, 4]:
        obs, reward, done, truncated, info = env.step(a)
        print(f"\n执行动作去装载点 {a} 后:")
        print("当前状态:", obs)
        print("获得奖励 (负数代表惩罚):", reward)