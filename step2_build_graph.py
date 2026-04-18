import json

json_path = "./conf/north_pit_mine.json" 

with open(json_path, "r", encoding="utf-8") as f:
    mine_data = json.load(f)

# 1. 解密 Road 字典
roads_data = mine_data.get('road', {})
print(f"道路数据的类型是: {type(roads_data)}")
print(f"一共有 {len(roads_data)} 条道路记录\n")

if isinstance(roads_data, dict):
    print("--- 道路 (Road) 字典的前 2 个元素示例 ---")
    # 遍历字典，打印前两个看看结构
    count = 0
    for road_id, road_info in roads_data.items():
        print(f"路段标识 (Key): {road_id}")
        print(f"路段属性 (Value): {road_info}\n")
        count += 1
        if count >= 2: 
            break

# 2. 统计卡车总数 (为强化学习做准备)
total_trucks = 0
truck_types = mine_data.get('charging_site', {}).get('trucks', [])
for t in truck_types:
    total_trucks += t['count']
print(f"--- 卡车统计 ---")
print(f"我们一共拥有 {total_trucks} 辆卡车可以用于调度！")