import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from step4_physics_env import PhysicsMineEnv
import os

# --- 0. 配置与参数设置 ---
# 如果你的代码不在这个目录下运行，请确保这里使用的是绝对路径或正确的相对路径
script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()

# 检查素材文件是否存在，防止报错
material_paths = {
    'shovel': os.path.join(script_dir, 'materials', '挖机.png'),
    'dump': os.path.join(script_dir, 'materials', 'dump.png'),
    'truck_empty': os.path.join(script_dir, 'materials', 'truck_unhauling.png'),
    'truck_full': os.path.join(script_dir, 'materials', 'truck_hauling.png')
}
for name, path in material_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到素材文件: {path}。请确保 materials 文件夹存在且包含对应的图片。")

img_shovel = mpimg.imread(material_paths['shovel'])
img_dump = mpimg.imread(material_paths['dump'])
img_truck_empty = mpimg.imread(material_paths['truck_empty'])
img_truck_full = mpimg.imread(material_paths['truck_full'])

SCALE_SHOVEL = 0.08
SCALE_DUMP = 0.08
SCALE_TRUCK = 0.04

# --- 1. 加载数据与环境 ---
# 请确保 JSON 配置文件路径正确
config_path = "/Users/qiantao/大三下/RL/conf/north_pit_mine.json" 
with open(config_path, "r", encoding="utf-8") as f:
    data = json.load(f)

load_coords = np.array([ls['position'] for ls in data['load_sites']])
dump_coords = np.array([ds['position'] for ds in data['dump_sites']])

# 初始化环境 (设置你需要的车辆数，这里假设与你原来代码一致，比如 25 辆)
env = PhysicsMineEnv(config_path=config_path, total_trucks=25)

# 设置兼容 Mac 的中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams["axes.unicode_minus"] = False

# --- 2. 坐标计算函数 ---
def get_visual_pos(truck):
    """根据 T=0 时的随机 timer 计算坐标"""
    state = truck['state']
    site_id = truck['site']
    timer = truck['timer']
    
    # 假设如果车辆正在路上 (state 1 或 3)，它的起点 origin_site 是随机分配的
    # (因为这是初始状态，我们通过 env.reset() 后手动补充一个合理的 origin)
    origin = truck.get('origin_site', np.random.randint(0, 5)) 
    
    if state in [0, 4]: 
        base_pos = dump_coords[site_id]
    elif state == 2: 
        base_pos = load_coords[site_id]
    elif state == 1: # 去装载 (空车)
        start_pos = dump_coords[origin] 
        end_pos = load_coords[site_id]
        total_time = max(0.1, env.d2l_matrix[origin][site_id])
        # progress: timer 越小越接近终点
        progress = np.clip(1 - (timer / total_time), 0, 1)
        base_pos = start_pos + (end_pos - start_pos) * progress
    else: # state == 3: 去卸载 (满车)
        start_pos = load_coords[origin] 
        end_pos = dump_coords[site_id]  
        total_time = max(0.1, env.l2d_matrix[origin][site_id])
        progress = np.clip(1 - (timer / total_time), 0, 1)
        base_pos = start_pos + (end_pos - start_pos) * progress
    
    # 加入一点点随机抖动，防止在同一个站点的多辆车完全重叠在一起看不清
    jitter = np.random.normal(0, 0.015, size=2)
    return base_pos + jitter

# --- 3. 绘制静态图像 ---
def render_static_env():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#F4F6F9') 
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("OpenMines - 初始环境拓扑与车辆分布 (T=0)", fontsize=18, pad=20, fontweight='bold')
    ax.axis('off')

    # 1. 重置环境，获取 T=0 的状态
    env.reset()
    
    # 关键修复：由于是初始随机状态，在途车辆没有真实的出发点，手动为它们分配一个起点用于连线计算
    for t in env.trucks:
        t['state'] = 0  # 0 表示空闲待命
        # 可以让它们集中在 0 号卸载点，也可以随机平均分配到各个卸载点
        t['site'] = np.random.randint(0, env.num_dumps) 
        t['timer'] = 0
        t['origin_site'] = t['site']
        
    # 清空可能因为随机初始化产生的排队数据
    env.load_queues = np.zeros(env.num_loads)
    env.dump_queues = np.zeros(env.num_dumps)

    # 2. 绘制装载点 (铲位)
    for i, coord in enumerate(load_coords):
        im = OffsetImage(img_shovel, zoom=SCALE_SHOVEL)
        ab = AnnotationBbox(im, coord, frameon=False, zorder=10)
        ax.add_artist(ab)
        # T=0 时的队列长度
        q_len = int(env.load_queues[i]) if hasattr(env, 'load_queues') else 0
        ax.text(coord[0], coord[1]-0.05, f'铲位 {i}\nQ: {q_len}', ha='center', fontsize=10, color='#5D4037', fontweight='bold')

    # 3. 绘制卸载点
    for i, coord in enumerate(dump_coords):
        im = OffsetImage(img_dump, zoom=SCALE_DUMP)
        ab = AnnotationBbox(im, coord, frameon=False, zorder=10)
        ax.add_artist(ab)
        q_len = int(env.dump_queues[i]) if hasattr(env, 'dump_queues') else 0
        ax.text(coord[0], coord[1]-0.05, f'卸载位 {i}\nQ: {q_len}', ha='center', fontsize=10, color='#263238', fontweight='bold')

    # 4. 绘制所有车辆
    for t in env.trucks:
        pos = get_visual_pos(t)
        # 根据状态判断是满车还是空车
        if t['state'] in [3, 4]: # 3(去卸载), 4(卸载中) 是满的
            im = OffsetImage(img_truck_full, zoom=SCALE_TRUCK)
        else: # 0(待命), 1(去装载), 2(装载中) 是空的
            im = OffsetImage(img_truck_empty, zoom=SCALE_TRUCK)
        
        ab = AnnotationBbox(im, pos, frameon=False, zorder=20)
        ax.add_artist(ab)
        
    # 5. 添加系统状态文本框
    info_text = f"SYSTEM INIT\nTime: 0.0 MIN\nTotal Trucks: {env.total_trucks}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontweight='bold', family='monospace', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#CCCCCC', boxstyle='round,pad=0.6'))

    # 6. 保存并展示图片
    output_filename = "environment_snapshot_T0.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 环境快照已生成并保存为：{output_filename}")
    plt.show()

# 运行主函数
if __name__ == "__main__":
    render_static_env()