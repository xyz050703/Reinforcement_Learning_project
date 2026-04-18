import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PhysicsMineEnv(gym.Env):
    def __init__(self, config_path="./conf/north_pit_mine.json"):
        super(PhysicsMineEnv, self).__init__()
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.mine_data = json.load(f)
            
        self.l2d_matrix = np.array(self.mine_data['road']['l2d_road_matrix'])
        self.d2l_matrix = np.array(self.mine_data['road']['d2l_road_matrix'])
        
        self.num_loads = len(self.l2d_matrix)
        self.num_dumps = len(self.l2d_matrix[0])
        self.total_trucks = 70
        
        self.action_space = spaces.Discrete(self.num_loads)
        self.observation_space = spaces.Box(
            low=0, high=self.total_trucks, shape=(21,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim_time = 0
        self.trucks = []
        self.ore_produced_this_step = 0 
        self.last_raw_reward = 0.0  # 【VecNormalize 改造】：用于记录归一化前的真实奖励，供图表使用
        
        for i in range(self.total_trucks):
            state = np.random.choice([0, 1, 2, 3, 4])
            l_idx = np.random.randint(0, self.num_loads)
            d_idx = np.random.randint(0, self.num_dumps)
            site = l_idx if state in [1, 2] else d_idx
            
            if state in [1, 3]: 
                max_t = self.d2l_matrix[0][l_idx] if state == 1 else self.l2d_matrix[l_idx][0]
                timer = np.random.uniform(0, max_t)
            elif state == 2: timer = np.random.uniform(0, 3.0) 
            elif state == 4: timer = np.random.uniform(0, 2.0) 
            else: timer = 0 
            
            self.trucks.append({
                'id': i, 'state': state, 'site': site, 'timer': timer
            })

        self.load_queues = np.zeros(self.num_loads, dtype=np.float32)
        self.dump_queues = np.zeros(self.num_dumps, dtype=np.float32)
        
        for t in self.trucks:
            if t['state'] == 2: self.load_queues[t['site']] += 1
            if t['state'] == 4: self.dump_queues[t['site']] += 1

        self.current_truck = self._fast_forward_time()
        if self.current_truck is None:
            self.current_truck = {'site': 0, 'state': -1}
            
        return self._get_obs(), {}

    def _get_obs(self):
        load_enroute = np.zeros(self.num_loads, dtype=np.float32)
        dump_enroute = np.zeros(self.num_dumps, dtype=np.float32)
        
        for t in self.trucks:
            if t['state'] == 1:   
                load_enroute[t['site']] += 1
            elif t['state'] == 3: 
                dump_enroute[t['site']] += 1
                
        curr_site = self.current_truck['site'] if self.current_truck else 0
                
        obs = [curr_site] + \
              list(self.load_queues) + list(self.dump_queues) + \
              list(load_enroute) + list(dump_enroute)
              
        return np.array(obs, dtype=np.float32)

    def _compute_real_time_state(self):
        load_q = np.zeros(self.num_loads, dtype=np.float32)
        dump_q = np.zeros(self.num_dumps, dtype=np.float32)
        load_enroute = np.zeros(self.num_loads, dtype=np.float32)
        dump_enroute = np.zeros(self.num_dumps, dtype=np.float32)
        
        for t in self.trucks:
            if t['state'] == 1:
                if t['timer'] <= 0: load_q[t['site']] += 1
                else: load_enroute[t['site']] += 1
            elif t['state'] == 2:
                load_q[t['site']] += 1
            elif t['state'] == 3:
                if t['timer'] <= 0: dump_q[t['site']] += 1
                else: dump_enroute[t['site']] += 1
            elif t['state'] == 4:
                dump_q[t['site']] += 1
                
        return load_q, dump_q, load_enroute, dump_enroute

    def _fast_forward_time(self):
        while self.sim_time < 480: 
            for t in self.trucks:
                if t['state'] == 0:
                    return t
            
            active_loading_sites = {t['site'] for t in self.trucks if t['state'] == 2}
            active_dumping_sites = {t['site'] for t in self.trucks if t['state'] == 4}

            for t in self.trucks:
                if t['timer'] > 0:
                    t['timer'] -= 1
                else:
                    if t['state'] == 1:   
                        if t['site'] not in active_loading_sites:
                            t['state'] = 2
                            t['timer'] = 3    
                            active_loading_sites.add(t['site']) 
                            
                    elif t['state'] == 2: 
                        t['state'] = 3
                        _, dump_q, _, _ = self._compute_real_time_state()
                        target_dump = int(np.argmin(dump_q)) 
                        active_loading_sites.discard(t['site']) 
                        t['timer'] = self.l2d_matrix[t['site']][target_dump]
                        t['site'] = target_dump
                        
                    elif t['state'] == 3: 
                        if t['site'] not in active_dumping_sites:
                            t['state'] = 4
                            t['timer'] = 2    
                            active_dumping_sites.add(t['site'])
                            
                    elif t['state'] == 4: 
                        t['state'] = 0
                        self.ore_produced_this_step += 1 
                        active_dumping_sites.discard(t['site'])
                        
            self.sim_time += 1
            
        return {'site': 0, 'state': -1} 

    def step(self, action):
        target_load_site = int(action)
        start_site = int(self.current_truck['site']) if self.current_truck else 0
        
        self.ore_produced_this_step = 0 
        
        travel_time = self.d2l_matrix[start_site][target_load_site]
        if self.current_truck:
            self.current_truck['state'] = 1
            self.current_truck['site'] = target_load_site
            self.current_truck['timer'] = travel_time
        
        load_q, dump_q, load_enroute, dump_enroute = self._compute_real_time_state()
        future_pressure = load_q[target_load_site] + load_enroute[target_load_site]
        
        self.load_queues = load_q
        self.dump_queues = dump_q
        
        next_truck = self._fast_forward_time()
        done = self.sim_time >= 480
        
        if next_truck is not None:
            self.current_truck = next_truck
        else:
            self.current_truck = {'site': 0, 'state': -1}
        # --- 【彻底重构的 Reward 逻辑】 ---
        
        # 1. 产量是硬道理：每产生一车矿，给予高额奖励
        production_reward = self.ore_produced_this_step * 50.0
        
        # 2. 全局拥堵税：只要系统里有车排队，每分钟都扣分！
        # 这样 AI 如果故意造堵车，它在整个 480 分钟里每分钟都会持续扣分，直接倒扣到破产
        total_current_queue = np.sum(load_q) + np.sum(dump_q)
        congestion_tax = -float(total_current_queue) * 0.5 
        
        # 3. 活跃奖励（可选）：鼓励车子跑在路上，而不是停在原地
        total_enroute = np.sum(load_enroute) + np.sum(dump_enroute)
        active_bonus = float(total_enroute) * 0.1
        
        # 最终 Reward
        reward = production_reward + congestion_tax + active_bonus
        
        self.last_raw_reward = float(reward) 
        
        return self._get_obs(), float(reward), done, False, {}