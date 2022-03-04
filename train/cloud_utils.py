import pandas as pd
from math import log10

class CloudCalculater:
    """
    根据6个云节点数据计算出2个最大区间节点：云底高度、云顶高度
    """
    @staticmethod
    def CalculateCloudIntervalHeight(cloud_List):
        cloud_bottom = 0
        cloud_top = 0
        i = 0
        while i < len(cloud_List):
            if cloud_List[i] and not cloud_bottom:
                cloud_bottom = cloud_List[i]
            i += 2

        cloud_top = cloud_bottom + cloud_List[1] + cloud_List[3] + cloud_List[5]
        cloud_top = 10000 if cloud_top > 10000 else cloud_top

        return [cloud_bottom, cloud_top]

    """
    根据探空文件进行云层计算
    根据得到的湿度，判断是否入云，计算云层信息
    hei_temp_humi_list:原始探空高度层对应温度湿度列表
    return
    [
        [53, 2750.0, 54, 3000.0],
        [61, 4750.0, 63, 5250.0],
        [66, 6000.0, 77, 8750.0],
        [79, 9250.0, 82, 10000.0]
    ]
    """
    @staticmethod
    def CalculateCloud(height_temperature_humidity_list):
        humi_list = []
        rh_list = []
        for data in height_temperature_humidity_list:
            new_humidity = calculate_humidity(data[1], data[2])
            humi_list.append(new_humidity)
            rh_list.append(calculate_rh(data[0]))

        humi_list[3] = 99
        cloud_bottom = []
        cloud_top = []
        bottom_index = []
        top_index = []
        for index, humidity in enumerate(humi_list):

            if humidity >= rh_list[index] and humi_list[index - 1] < rh_list[
                    index - 1]:
                bottom = height_temperature_humidity_list[index][0]
                cloud_bottom.append(bottom)
                bottom_index.append(index)
            if len(cloud_bottom) > len(cloud_top):
                if height_temperature_humidity_list[index][
                        0] == height_temperature_humidity_list[-1][0]:
                    cloud_top.append(height_temperature_humidity_list[-1][0])
                    top_index.append(index)
                    continue

                if humidity < rh_list[index] and humi_list[
                        index - 1] >= rh_list[index - 1]:
                    top = height_temperature_humidity_list[index][0]
                    if top >= 500:
                        cloud_top.append(top)
                        top_index.append(index)
                    else:
                        cloud_bottom.pop()
                        bottom_index.pop()

        cloud_data = []
        for i in range(0, len(cloud_top)):

            cloud_data.append(
                [bottom_index[i], cloud_bottom[i], top_index[i], cloud_top[i]])

        todo_remove_list = []
        for one_data in cloud_data:
            # 云层判断
            if one_data[3] < one_data[1] and one_data[2] > one_data[0]:
                todo_remove_list.append(one_data)
                continue
            if one_data[3] - one_data[1] < 80:

                for i in range(one_data[0], one_data[2] + 1, 1):
                    if humi_list[i] < rh_list[i] + 3:
                        yc_flag = True
                    else:
                        yc_flag = False
                if yc_flag:
                    todo_remove_list.append(one_data)

        for one_data in todo_remove_list:
            cloud_data.remove(one_data)

        new_cloud_data = []

        yjc_flag = False
        x = 0
        for j in range(0, len(cloud_data), 1):
            if yjc_flag is False:
                new_cloud_data.append(cloud_data[j])
            # 云夹层判断
            if j == len(cloud_data) - 1:
                break
            if cloud_data[j +
                          1][1] - new_cloud_data[x][3] < 300 and cloud_data[
                              j + 1][1] > new_cloud_data[x][3]:

                for i in range(new_cloud_data[x][2] + 1, cloud_data[j + 1][0],
                               1):
                    # if humi_list[i] > rh_list[i] - 5:
                    if humi_list[i] > 5:
                        yjc_flag = True
                    else:
                        yjc_flag = False
                if yjc_flag:

                    new_cloud_data[x][2] = cloud_data[j + 1][2]
                    new_cloud_data[x][3] = cloud_data[j + 1][3]
                else:
                    x = x + 1
            else:
                yjc_flag = False
                x = x + 1

        return new_cloud_data

# 根据高度分类计算入云判定的湿度阈值
def calculate_rh(height):
    height_km = height / 1000
    if 0 <= height < 1000:
        rh = 91
    elif 1000 <= height < 2000:
        rh = -6.416 * height_km + 97
    elif 2000 <= height < 7562:
        rh = -1.223 * height_km + 87
    elif 7562 <= height < 10000:
        rh = -4 * height_km + 108
    else:
        rh = 68

    return rh

# 由每层的高度、湿度计算冰面饱和水汽压下的空气相对湿度
def calculate_humidity(temperature, humidity):
    T = temperature
    if T < 273.15:
        logEw = 10.79574 * (1 - 273.16 / T) - 5.028 * log10(T / 273.16) + \
                1.50475 * 0.0001 * (1 - 10 ** (-8.2969 * (T / 273.16 - 1))) + \
                0.42873 * 0.001 * (10 ** (4.76955 * (1 - 273.16 / T)) - 1) + 0.78614
        logEi = -9.09685 * (273.16 / T - 1) - 3.56654 * log10(
            273.16 / T) + 0.87682 * (1 - T / 273.16) + 0.78614
        Ur = humidity * ((10 ** logEw) / (10 ** logEi))
        humidity = Ur
    else:
        pass
    return humidity