# 使用指南
1. 需要训练的模型配置放置于config目录下的untrained目录中。
每个配置视为一个任务。
程序会批量训练untrained目录下所有的任务。
2. 配置文件中：
   1. forward_result_path和sounding_path为正演结果目录与探空目录，
   里面为训练需要用到的数据。
   2. model_name为输入的模型文件设置文件名。若文件名为空，则以默认名称保存。
   3. data_sources中的stime,etime为训练数据的时间筛选项。
   4. hidden_nodes为隐层节点个数范围。
3. untrained中的任务处理结束后，任务会移动至trained目录下。
4. 训练结果保存至out目录下的model目录中。
5. 日志文件在log目录中

# 部署指南
1. 使用pip安装依赖:
   pip install -r requirements
2. 运行可视化界面程序：
   python ./gui.py
