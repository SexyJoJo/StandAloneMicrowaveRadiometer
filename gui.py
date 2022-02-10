import os
import tkinter as tk

def set_config():
    top = tk.Toplevel(root)
    window_centered(top, 800, 600)
    top.title("新建训练任务")

    tk.Label(top, text="").grid(row=0, column=1)
    tk.Label(top, text="训练任务名称：", font=('Arial', 12)).grid(row=1, column=2)
    tk.Label(top, text="站台号：", font=('Arial', 12)).grid(row=2, column=2)
    tk.Label(top, text="数据起始时间：", font=('Arial', 12)).grid(row=3, column=2)
    tk.Label(top, text="数据结束时间：", font=('Arial', 12)).grid(row=4, column=2)
    tk.Label(top, text="云层扰动", font=('Arial', 12)).grid(row=5, column=2)
    tk.Label(top, text="K通道扰动", font=('Arial', 12)).grid(row=6, column=2)
    tk.Label(top, text="V通道扰动", font=('Arial', 12)).grid(row=7, column=2)
    tk.Label(top, text="弱吸收通道扰动", font=('Arial', 12)).grid(row=8, column=2)


def window_centered(window, width, height):
    """窗口默认居中并显示"""
    sw = window.winfo_screenwidth()  # 得到屏幕宽度
    sh = window.winfo_screenheight()  # 得到屏幕高度
    ww = width
    wh = height
    x = (sw - ww) / 2
    y = (sh - wh) / 2
    window.geometry("%dx%d+%d+%d" % (ww, wh, x, y))


root = tk.Tk()
root.title("单机训练程序")
window_centered(root, 800, 600)

task_label = tk.Label(root, text="待训练任务：", font=('Arial', 15))

create_button = tk.Button(root, text="新建", command=set_config)

task_list = tk.Listbox(root, width=80, height=25, font=('Arial', 12), selectmode="extended")
for task in os.listdir(r"config/untrained"):
    task_list.insert(0, task)

task_label.pack()
task_list.pack()
create_button.pack()
root.mainloop()
