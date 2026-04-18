import os
import csv
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback
from typing import Callable
from step4_physics_env import PhysicsMineEnv

# ==========================================
# 0. 全局实验配置
# ==========================================
WANDB_PROJECT = "Mine-Dispatch-Optimization-NEW"
BASE_SAVE_DIR = "./results/Fleet_Search_Experiment/"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ==========================================
# 1. 回调函数与辅助函数
# ==========================================
class MineUnifiedLogger(BaseCallback):
    """精简版的 Logger，主要用于 Wandb 上报"""
    def __init__(self, verbose=0):
        super(MineUnifiedLogger, self).__init__(verbose)
        self.current_ep_reward = 0
        self.current_ep_production = 0
        self.current_ep_queue_sum = 0
        self.current_ep_steps = 0

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].unwrapped
        self.current_ep_reward += env.last_raw_reward
        self.current_ep_production += getattr(env, 'ore_produced_this_step', 0)
        
        real_queue = np.sum(env.load_queues) + np.sum(env.dump_queues)
        self.current_ep_queue_sum += real_queue
        self.current_ep_steps += 1
        
        if self.locals["dones"][0]:
            avg_queue = self.current_ep_queue_sum / self.current_ep_steps if self.current_ep_steps > 0 else 0
            wandb.log({
                "mine_business/episode_total_reward": self.current_ep_reward,
                "mine_business/total_production": self.current_ep_production,
                "mine_business/avg_queue_length": avg_queue,
                "global_step": self.num_timesteps
            })
            self.current_ep_reward = 0
            self.current_ep_production = 0
            self.current_ep_queue_sum = 0
            self.current_ep_steps = 0
        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 2. 核心训练与评估包装函数
# ==========================================
def train_and_evaluate_fleet(fleet_size: int, total_timesteps: int = 1000000):
    """针对特定车辆数进行完整的 训练 -> 评估 -> 返回结果 流程"""
    experiment_name = f"PPO_FleetSize_{fleet_size}"
    save_path = os.path.join(BASE_SAVE_DIR, experiment_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"🚀 开始测试车辆规模: {fleet_size} 辆车")
    print(f"{'='*50}")

    # 1. 环境初始化
    env = DummyVecEnv([lambda: PhysicsMineEnv(config_path="./conf/north_pit_mine.json", total_trucks=fleet_size)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # 2. Wandb 初始化 (每个循环开启一个新 run)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=experiment_name,
        sync_tensorboard=True, 
        reinit=True # 允许在同一个脚本中多次 init
    )

    # 3. 创建 PPO 模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0, # 关掉底层的狂轰滥炸输出，保持控制台清爽
        learning_rate=linear_schedule(0.0003),    
        n_steps=4096,            
        batch_size=256,          
        n_epochs=10,             
        gamma=0.99,              
        ent_coef=0.005,           
        clip_range=0.2,          
        tensorboard_log=os.path.join(save_path, "tb_logs/") 
    )

    callbacks = CallbackList([
        WandbCallback(verbose=0), 
        MineUnifiedLogger()
    ])

    # 4. 开始训练
    print(f"正在进行 {total_timesteps} 步模型训练，请稍候...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)

    # 保存环境参数 (后续测试需要)
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    # 5. 最终效果评估 (跑 5 个班次取平均值)
    print(f"训练完成，正在进行最终性能评估...")
    env.training = False
    env.norm_reward = False
    
    eval_episodes = 5
    results_reward, results_production, results_queue = [], [], []

    for ep in range(eval_episodes):
        obs = env.reset()
        ep_reward, ep_production, ep_queue_sum, steps = 0, 0, 0, 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            base_eval_env = env.envs[0].unwrapped
            ep_reward += rewards[0]
            ep_production += base_eval_env.ore_produced_this_step
            ep_queue_sum += (np.sum(base_eval_env.load_queues) + np.sum(base_eval_env.dump_queues))
            steps += 1
            
            if dones[0]:
                results_reward.append(ep_reward)
                results_production.append(ep_production)
                results_queue.append(ep_queue_sum / steps)
                break

    # 结束当前的 wandb run
    run.finish()
    
    # 计算均值
    final_metrics = {
        "Fleet_Size": fleet_size,
        "Avg_Reward": round(np.mean(results_reward), 2),
        "Avg_Production": round(np.mean(results_production), 2),
        "Avg_Queue_Length": round(np.mean(results_queue), 2)
    }
    
    print(f"✅ {fleet_size} 辆车评估结果: 产量={final_metrics['Avg_Production']}, 拥堵={final_metrics['Avg_Queue_Length']}")
    return final_metrics


# ==========================================
# 3. 主循环与数据持久化
# ==========================================
if __name__ == "__main__":
    # 定义测试的车辆范围 (从 45 开始，到 55 结束，步长为 1)
    fleet_sizes_to_test = list(range(45, 56, 1))
    
    all_results = []
    csv_file_path = os.path.join(BASE_SAVE_DIR, "fleet_optimization_results.csv")
    
    # 写入 CSV 表头
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Fleet_Size", "Avg_Reward", "Avg_Production", "Avg_Queue_Length"])
        writer.writeheader()

    for fleet_size in fleet_sizes_to_test:
        # 执行一整套训练和评估
        metrics = train_and_evaluate_fleet(fleet_size=fleet_size, total_timesteps=1000000)
        all_results.append(metrics)
        
        # 每跑完一个规模，就立刻追加写入 CSV，防止断电或意外中断丢失数据
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writerow(metrics)

    print(f"\n🎉 所有实验运行完毕！")
    print(f"📊 汇总数据已保存至: {csv_file_path}")