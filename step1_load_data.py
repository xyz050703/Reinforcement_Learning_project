import json

# 注意确认路径是否正确
json_path = "/Users/qiantao/大三下/RL/conf/north_pit_mine.json" 

with open(json_path, "r", encoding="utf-8") as f:
    mine_data = json.load(f)

print("--- 1. 装载点 (Load Sites) 的第一个元素 ---")
if 'load_sites' in mine_data and len(mine_data['load_sites']) > 0:
    print(mine_data['load_sites'][0])

print("\n--- 2. 卸载点 (Dump Sites) 的第一个元素 ---")
if 'dump_sites' in mine_data and len(mine_data['dump_sites']) > 0:
    print(mine_data['dump_sites'][0])

print("\n--- 3. 充电站/停车场 (Charging Site) ---")
if 'charging_site' in mine_data:
    # 充电站可能是一个列表，也可能是一个字典，我们打印出来看看
    if isinstance(mine_data['charging_site'], list) and len(mine_data['charging_site']) > 0:
        print(mine_data['charging_site'][0])
    else:
        print(mine_data['charging_site'])

print("\n--- 4. 道路 (Road) 的第一个元素 ---")
if 'road' in mine_data and len(mine_data['road']) > 0:
    print(mine_data['road'][0])