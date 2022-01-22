from train.training import Train
from datetime import datetime
import os
import json
import shutil

if __name__ == '__main__':
    t1 = datetime.now()
    untrained_tasks = os.listdir(r"config/untrained")
    # 遍历待训练任务
    for untrained_task in untrained_tasks:
        untrained_path = os.path.join("config/untrained", untrained_task)
        trained_path = os.path.join("config/trained", untrained_task)
        with open(untrained_path, "r", encoding='utf8') as f:
            config = json.load(f)
        # 训练
        status = Train.training(config)

        # 训练成功后移动文件到已训练文件夹
        if status:
            shutil.move(untrained_path, trained_path)
    t2 = datetime.now()
    print(f"完成时间：{t2-t1}\n")
