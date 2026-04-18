import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
# 【VecNormalize 改造】：引入必要的 Wrapper 工具
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from step4_physics_env import PhysicsMineEnv
from typing import Callable

# ==========================================
# 0. 全局实验配置
# ==========================================
WANDB_PROJECT = "Mine-Dispatch-Optimization"
WANDB_EXPERIMENT = "PPO_70Cars_Baseline_v1_Normalized"  # 建议改个名字区分一下

SAVE_DIR = f"./results/{WANDB_PROJECT}/{WANDB_EXPERIMENT}/"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. 统一回调函数
# ==========================================
class MineUnifiedLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(MineUnifiedLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_productions = []
        self.episode_avg_queues = []
        
        self.current_ep_reward = 0
        self.current_ep_production = 0
        self.current_ep_queue_sum = 0
        self.current_ep_steps = 0

    def _on_step(self) -> bool:
        # 获取最底层的真实环境对象
        env = self.training_env.envs[0].unwrapped
        
        # 【VecNormalize 改造】：不再使用 locals["rewards"][0]，因为它被归一化了。改用我们保存的真实 Reward
        self.current_ep_reward += env.last_raw_reward
        self.current_ep_production += getattr(env, 'ore_produced_this_step', 0)
        
        real_queue = np.sum(env.load_queues) + np.sum(env.dump_queues)
        self.current_ep_queue_sum += real_queue
        self.current_ep_steps += 1
        
        # VecEnv 的 done 是一个数组
        done = self.locals["dones"][0]
        if done:
            avg_queue = self.current_ep_queue_sum / self.current_ep_steps if self.current_ep_steps > 0 else 0
            
            wandb.log({
                "mine_business/episode_total_reward": self.current_ep_reward,
                "mine_business/total_production": self.current_ep_production,
                "mine_business/avg_queue_length": avg_queue,
                "global_step": self.num_timesteps
            })
            
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_productions.append(self.current_ep_production)
            self.episode_avg_queues.append(avg_queue)
            
            self.current_ep_reward = 0
            self.current_ep_production = 0
            self.current_ep_queue_sum = 0
            self.current_ep_steps = 0
            
        return True

# ==========================================
# 2. 辅助函数
# ==========================================
def plot_and_save_metrics(logger, save_path, window_size=50):
    def moving_average(data, window_size):
        if len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    episodes = np.arange(len(logger.episode_rewards))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    titles = ['Episode Total Reward', 'Episode Total Production (Ore)', 'Episode Avg Queue Length']
    y_labels = ['Reward', 'Production', 'Avg Vehicles in Queue']
    data_lists = [logger.episode_rewards, logger.episode_productions, logger.episode_avg_queues]

    for i in range(3):
        data = data_lists[i]
        color = colors[i]
        axs[i].plot(episodes, data, color=color, alpha=0.3, label='Raw Data')
        
        smoothed_data = moving_average(data, window_size)
        if len(smoothed_data) > 0:
            axs[i].plot(np.arange(window_size-1, len(data)), smoothed_data, 
                        color=color, linewidth=2, label=f'Trend (MA-{window_size})')
            
        axs[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axs[i].set_xlabel('Episodes (Shifts)', fontsize=12)
        axs[i].set_ylabel(y_labels[i], fontsize=12)
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    image_file = os.path.join(save_path, f"{WANDB_EXPERIMENT}_metrics.png")
    plt.savefig(image_file, dpi=300)
    print(f"\n✅ 本地训练图像已保存至: {image_file}")
    plt.close()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 主流程代码
# ==========================================
if __name__ == "__main__":
    print("1. 初始化自定义矿区环境...")
    # 先验证原生环境
    base_env = PhysicsMineEnv(config_path="./conf/north_pit_mine.json")
    check_env(base_env)

    # 【VecNormalize 改造】：包装成向量环境，并应用归一化
    # 注意：DummyVecEnv 需要传入一个函数列表
    env = DummyVecEnv([lambda: PhysicsMineEnv(config_path="./conf/north_pit_mine.json")])
    # 状态和奖励同时归一化，截断值设为 10（防止极端异常值带偏均值）
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    ppo_config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,   # 先跑 100 万步看看收敛趋势
        "learning_rate": linear_schedule(0.0003),
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.005,            # 降低了随机探索惩罚
        "clip_range": 0.2,
    }
    
    print(f"2. 初始化 Wandb (项目: {WANDB_PROJECT}, 实验: {WANDB_EXPERIMENT})...")
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_EXPERIMENT,
        config=ppo_config,
        sync_tensorboard=True, 
        monitor_gym=True,
        save_code=True,
    )

    print("3. 创建 PPO 强化学习智能体...")
    model = PPO(
        ppo_config["policy_type"], 
        env, 
        verbose=1, 
        learning_rate=ppo_config["learning_rate"],    
        n_steps=ppo_config["n_steps"],            
        batch_size=ppo_config["batch_size"],          
        n_epochs=ppo_config["n_epochs"],             
        gamma=ppo_config["gamma"],              
        ent_coef=ppo_config["ent_coef"],           
        clip_range=ppo_config["clip_range"],          
        tensorboard_log=os.path.join(SAVE_DIR, "tensorboard_logs/") 
    )

    wandb_cb = WandbCallback(
        gradient_save_freq=100000,
        model_save_path=os.path.join(SAVE_DIR, "checkpoints/"),
        verbose=2,
    )
    unified_cb = MineUnifiedLogger()
    callbacks = CallbackList([wandb_cb, unified_cb])

    print("4. 开始极限训练...")
    model.learn(total_timesteps=ppo_config["total_timesteps"], progress_bar=True, callback=callbacks)

    print("5. 训练完成，正在保存数据和模型...")
    final_model_path = os.path.join(SAVE_DIR, f"{WANDB_EXPERIMENT}_final_model")
    model.save(final_model_path)
    
    # 【VecNormalize 改造】：千万记得保存环境参数！
    stats_path = os.path.join(SAVE_DIR, "vec_normalize.pkl")
    env.save(stats_path)
    print(f"✅ 模型已保存至: {final_model_path}.zip")
    print(f"✅ 归一化参数已保存至: {stats_path}")
    
    plot_and_save_metrics(unified_cb, save_path=SAVE_DIR, window_size=50)
    run.finish()

    print("\n6. 智能体效果最终测试...")
    # 【VecNormalize 改造】：测试时必须加载训练出的均值和方差，并关闭状态更新
    eval_env = DummyVecEnv([lambda: PhysicsMineEnv(config_path="./conf/north_pit_mine.json")])
    eval_env = VecNormalize.load(stats_path, eval_env)
    
    # 核心设定：测试时，不更新均值，不缩放 Reward（以便看到真实业务得分）
    eval_env.training = False
    eval_env.norm_reward = False

    # VecEnv 的 reset 只返回 obs 数组
    obs = eval_env.reset()
    
    total_reward = 0
    total_real_queue = 0
    total_production = 0 
    steps = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        # VecEnv 的 step 返回 4 个值 (没有 truncated)
        obs, rewards, dones, infos = eval_env.step(action)
        
        # 底层环境需要通过 unwrapped 获取
        base_eval_env = eval_env.envs[0].unwrapped
        
        # rewards 此时是未归一化的真实分值（因为 norm_reward=False）
        total_reward += rewards[0]
        real_queue = np.sum(base_eval_env.load_queues) + np.sum(base_eval_env.dump_queues)
        total_real_queue += real_queue
        total_production += base_eval_env.ore_produced_this_step
        steps += 1
        
        # VecEnv 的 done 是一个列表/数组，比如 [True]
        if dones[0]:
            print(f"\n====================================")
            print(f"测试班次 (8小时) 结束！")
            print(f"总调度决策次数: {steps}")
            print(f"【AI 得分 (Reward)】: {total_reward:.2f}")
            print(f"【总拉矿量】: {total_production} 车")
            print(f"【平均排队车辆】: {total_real_queue/steps:.2f} 辆")
            print(f"====================================")
            break