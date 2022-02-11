from ast import arg
import os
import tkinter as tk
from tkinter import *
import json
import uuid
from parse.file_utils import FileUtils

# 模型文件路径
paramsJsonPath = "./config/untrained/"

def window_centered(window, width, height):
    """窗口默认居中并显示"""
    sw = window.winfo_screenwidth()  # 得到屏幕宽度
    sh = window.winfo_screenheight()  # 得到屏幕高度
    ww = width
    wh = height
    x = (sw - ww) / 2
    y = (sh - wh) / 2
    window.geometry("%dx%d+%d+%d" % (ww, wh, x, y))

class Application(tk.Frame):
    def __init__(self, master=None, queue = None):
        super().__init__(master)
        # 消息队列
        self.queue = queue
        self.master = master
        self.pack()
        self.create_widgets()

    def flushQueue(self):
        self.task_list.delete(0, END) 
        for task in os.listdir(r"config/untrained"):
            self.task_list.insert(END, task)

    def create_widgets(self):

        self.lable = tk.Label(self, text="待训练任务：", font=('Arial', 15))
        self.lable.pack()
        self.task_list = tk.Listbox(self, width=80, height=25, font=('Arial', 12), selectmode="extended")
        # 加载队列数据
        self.flushQueue()
        self.task_list.pack()

        self.create = tk.Button(self)
        self.create["text"] = "新建训练任务"
        self.create["command"] = self.createTask
        self.create.pack(side="bottom")

        # self.quit = tk.Button(self, text="QUIT", fg="red",
        #                       command=self.master.destroy)
        # self.quit.pack(side="bottom")

    # 创建训练任务GUI
    def createTask(self):

        self.top = tk.Tk()
        window_centered(self.top, 800, 600)
        self.top.title("新建训练任务")

        tk.Label(self.top, text="训练任务名称：", font=('Arial', 12)).grid(row=1, column=2)
        tk.Label(self.top, text="站台号：", font=('Arial', 12)).grid(row=2, column=2)
        tk.Label(self.top, text="数据起始时间：", font=('Arial', 12)).grid(row=3, column=2)
        tk.Label(self.top, text="数据结束时间：", font=('Arial', 12)).grid(row=4, column=2)
        tk.Label(self.top, text="云层扰动(%)", font=('Arial', 12)).grid(row=5, column=2)
        tk.Label(self.top, text="K通道扰动", font=('Arial', 12)).grid(row=6, column=2)
        tk.Label(self.top, text="V通道扰动", font=('Arial', 12)).grid(row=7, column=2)
        tk.Label(self.top, text="弱吸收通道扰动", font=('Arial', 12)).grid(row=8, column=2)

        model_name = tk.Entry(self.top)
        model_name.grid(row=1, column=3)
        sounding_station_id = tk.Entry(self.top)
        sounding_station_id.grid(row=2, column=3)
        stime = tk.Entry(self.top)
        stime.grid(row=3, column=3)
        etime = tk.Entry(self.top)
        etime.grid(row=4, column=3)
        cloud_disturb = tk.Entry(self.top)
        cloud_disturb.grid(row=5, column=3)
        k_disturb = tk.Entry(self.top)
        k_disturb.grid(row=6, column=3)
        v_disturb = tk.Entry(self.top)
        v_disturb.grid(row=7, column=3)
        absorb_disturb = tk.Entry(self.top)
        absorb_disturb.grid(row=8, column=3)
        
        button = tk.Button(self.top, text = "确定" , 
            command=lambda :self.commit(model_name,sounding_station_id,
            stime,etime,cloud_disturb,k_disturb,v_disturb,absorb_disturb))
        button.grid(column = 59, row = 1)
        #   command=self.master.destroy)

    # 用户提交
    def commit(self, model_name,sounding_station_id,
            stime,etime,cloud_disturb,k_disturb,v_disturb,absorb_disturb):

        dataSource = {
            "stime": stime.get(),
            "etime": etime.get(),
            "cloud_disturb": cloud_disturb.get(),
            "k_disturb": k_disturb.get(),
            "v_disturb": v_disturb.get(),
            "absorb_disturb" : absorb_disturb.get()
        }

        # 加载模板文件
        with open("./config/template.json", 'r') as file:
            file_dict = json.load(file)

        file_dict["model_name"] = model_name.get()
        file_dict["sounding_station_id"] = sounding_station_id.get()
        file_dict["data_sources"].append(dataSource)
        fileName = model_name.get() + "_" + str(uuid.uuid1())
        
        # 生成json文件
        with open(paramsJsonPath + fileName + ".json", 'w', encoding='utf8') as parameters_file:
            json.dump(file_dict, parameters_file, ensure_ascii=False, indent=4)

        # 加入缓存队列
        product(self.queue, fileName, self)


# 创建主界面
def mainGUI(q):
    root = tk.Tk()
    root.title("单机训练程序")
    window_centered(root, 800, 600)
    app = Application(master=root,queue = q)

    # 创建消费者线程
    t2=threading.Thread(target=consume,args=(q,app,))
    t2.start()

    # 主线程占用进入事件（消息）循环 -主线程阻塞
    app.mainloop()


from queue import Queue
import time,threading

def product(q, name, app):
    q.put('{}'.format(name)) #存放json文件路径到队列
    print ('新增：训练任务-{}'.format(name))
    # 刷新队列页面
    app.flushQueue()

def consume(q,app):
    training = 0 # 当前是否有训练任务标记
    while True:
        if not training and not q.empty():
            training = 1
            task = q.get()
            print ('训练模型：{}'.format(task)) #如果队列为空，则阻塞在这里持续等待队列中有任务
            # TODO 进行训练
            time.sleep(6)    
            q.task_done()
            # 删除相应json文件
            FileUtils.DeleteFile(paramsJsonPath + task + ".json")
            # 刷新队列页面
            app.flushQueue()
            training = 0


if __name__ == '__main__':
    
    # 消息队列
    q=Queue()
    # 生产者主进程
    mainGUI(q)


