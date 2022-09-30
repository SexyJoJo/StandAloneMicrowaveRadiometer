import copy
import numpy as np
from const.Consts import DEVICE_INFO, LIQUID_INFO, STANDARD_CHANNELS
from parse.parse_utils import *
from parse.file_utils import FileUtils


def liquid_params(station, forwardResults, allLv1Datas, threshold=10):
    """生成液态水订正参数"""
    # 初始化
    dataTemplate = {
        "band": "22.24GHz",
        # 亮温值
        "temp": 74.71968,
        "times": 1,
        # △T：各通道设备实测亮温T测与探空模拟亮温Tk的差值 or t_liquid
        # t_liquid:表征液态水通道亮温
        "type": "△T"
    }

    # 各个通道液态水时序列表
    liquidTimesList = []
    # 通道频率列表
    frequencyList = []
    # 各个通道表征液态水含量
    tLiquidList = {}
    # 各个通道的△T
    tDiffList = {}

    station = str(station)
    validChannelNumbers = DEVICE_INFO[station]["channels_map"]
    if DEVICE_INFO[station]["factory"] not in LIQUID_INFO.keys():
        liquid_channel_index = 7
    else:
        liquid_channel_index = LIQUID_INFO[DEVICE_INFO[station]["factory"]]["liquid_channel_index"]

    for number in validChannelNumbers:
        tLiquidList[STANDARD_CHANNELS[number]] = []
        tDiffList[STANDARD_CHANNELS[number]] = []

    for i, number in enumerate(validChannelNumbers):
        times = 1  # 时次记录
        for forwardTime, forwardBt in forwardResults.items():
            isMatch = False
            # 该时刻模拟亮温数据
            sd_bt_dict = {}
            # 该时刻实测亮温数据
            mr_bt_dict = {}
            # 该时刻表征液态水含量
            t_liquid_dict = {}
            # 该时刻的△T：每个时刻的各通道设备实测亮温T测与探空模拟亮温Tk的差值
            t_diffs_dict = {}
            for lv1Time, lv1Bt in allLv1Datas.items():
                if forwardTime == lv1Time:
                    isMatch = True
                    # 设备实测亮温
                    data = copy.deepcopy(dataTemplate)
                    data["band"] = STANDARD_CHANNELS[number] + "GHz"
                    data["temp"] = lv1Bt[i]
                    data["times"] = times
                    data["type"] = 'mr'
                    mr_bt_dict = data

                    # 正演模拟亮温
                    data = copy.deepcopy(dataTemplate)
                    data["band"] = STANDARD_CHANNELS[number] + "GHz"
                    # 是否有云只判断正演结果时刻
                    data["temp"] = forwardBt[i]
                    data["times"] = times
                    data["type"] = 'sd'
                    sd_bt_dict = data

                    liquidTemp = lv1Bt[liquid_channel_index]

                    # 复用其他不变的字典数据
                    t_liquid_dict = copy.deepcopy(mr_bt_dict)
                    t_liquid_dict["temp"] = liquidTemp
                    t_liquid_dict["type"] = "t_liquid"
                    tLiquidList[STANDARD_CHANNELS[number]].append(
                        t_liquid_dict["temp"])

                    t_diffs_dict = copy.deepcopy(sd_bt_dict)
                    t_diffs_dict["temp"] = round(mr_bt_dict["temp"] - sd_bt_dict["temp"], 3)
                    t_diffs_dict["type"] = "△T"
                    tDiffList[STANDARD_CHANNELS[number]].append(
                        t_diffs_dict["temp"])

                    # 时刻找到, 结束匹配
                    break

            if not isMatch:
                print("正演时刻结果未匹配到lv1数据", forwardTime)
            else:
                """
                剔除数据
                偏差阈值剔除的时候不考虑弱吸收通道，其他通道照常剔除
                35-54.5GHz为弱吸收通道扰动
                """
                isDelete = False
                # 剔除条件：非弱吸收通道、阈值不为0、模拟亮温与实测亮温的差值大于所选阈值

                if float(threshold) != 0 and abs(
                        float(t_liquid_dict["temp"]) -
                        float(t_diffs_dict["temp"])) > float(threshold):
                    isDelete = True
                    # print("剔除通道:", STANDARD_CHANNELS[number], "时刻：",
                    #       forwardTime)
                # 不剔除
                if not isDelete:
                    liquidTimesList.append(t_diffs_dict)
                    liquidTimesList.append(t_liquid_dict)
                    times += 1

        frequencyList.append(STANDARD_CHANNELS[number])

    # 各个频率通道平均值以及拟合曲线参数
    liquidParams = []
    dataTemplate = {
        "band": "22.24Ghz",
        "t_liquid_avg": 76.29,
        "reg": [0.96, 1.46],
        "△T_avg": 74.62
    }

    # 组织各个频率通道平均值以及拟合曲线参数-----------------------------------------------
    uncorr_channels = []  # 无需订正的通道
    for frequency in frequencyList:
        if len(tDiffList[frequency]) == 0 or len(
                tLiquidList[frequency]) == 0:
            continue
        data = copy.deepcopy(dataTemplate)
        data["band"] = str(frequency) + "GHz"
        if 54 <= float(frequency) <= 60 or frequency == list(tLiquidList.keys())[liquid_channel_index]:
            data["reg"] = [0, 0, 0]
            uncorr_channels.append(str(frequency) + 'GHz')
        else:
            correction = np.polyfit(x=tLiquidList[frequency],
                                    y=tDiffList[frequency],
                                    deg=2)
            data["reg"] = [round(correction[0], 3), round(correction[1], 2), round(correction[2], 2)]
        liquidParams.append(data["reg"])

    a, b, c = [], [], []
    for ch_params in liquidParams:
        a.append(ch_params[0])
        b.append(ch_params[1])
        c.append(ch_params[2])

    return [{"coeffs": [a, b, c], "apply": "a*x²+bx+c", "condition": ""}]


def reg_params(station, forwardResults, allLv1Datas, threshold=10, liqParams=None):
    """生成偏差订正参数"""
    # 各个通道亮温时序列表
    btTimesList = []

    dataTemplate = {
        "band": "22.24GHz",
        "condition": "sunny",
        "obs_time": "2018-08-01 01:00",
        # 亮温值
        "temp": 74.71968,
        "times": 1,
        # sd:正演模拟亮温 sounding
        # mr:设备亮温实测 measure
        "type": "sd"
    }

    # 读取液态水订正系数
    if liqParams:
        liq_reg_a = liqParams[0]['coeffs'][0]
        liq_reg_b = liqParams[0]['coeffs'][1]
        liq_reg_c = liqParams[0]['coeffs'][2]

    # 通道频率列表
    frequencyList = []
    # 各个通道设备实测亮温
    lv1BtList = {}
    # 各个通道正演模拟亮温
    forwardBtList = {}
    # 超过阈值需要剔除的时次 剔除：剔除所有通道该时刻的数据
    removeTime = []

    station = str(station)
    validChannelNumbers = DEVICE_INFO[station]["channels_map"]
    if DEVICE_INFO[station]["factory"] not in LIQUID_INFO.keys():
        liquid_channel_index = 7
    else:
        liquid_channel_index = LIQUID_INFO[DEVICE_INFO[station]["factory"]]["liquid_channel_index"]

    for number in validChannelNumbers:
        lv1BtList[STANDARD_CHANNELS[number]] = []
        forwardBtList[STANDARD_CHANNELS[number]] = []

    for index, number in enumerate(validChannelNumbers):
        times = 1
        for forwardTime, forwardBt in forwardResults.items():
            if forwardTime in removeTime:
                # print("异常时刻-剔除通道:", standChannel.frequency, "时刻：", forwardResult.datetime)
                continue

            isMatch = False

            # 该时刻模拟亮温数据
            sd_bt_dict = {}
            # 该时刻实测亮温数据
            mr_bt_dict = {}
            # 匹配lv1实测亮温
            for lv1Time, lv1Bt in allLv1Datas.items():
                if forwardTime == lv1Time:
                    isMatch = True

                    liquidTemp = lv1Bt[liquid_channel_index]

                    # 设备实测亮温
                    data = copy.deepcopy(dataTemplate)
                    data["band"] = STANDARD_CHANNELS[number] + "GHz"
                    # 是否有云只判断正演结果时刻
                    data["temp"] = round(lv1Bt[index], 3)
                    if liqParams:
                        data["temp"] = data["temp"] - round((
                                liq_reg_a[index] * liquidTemp * liquidTemp +
                                liq_reg_b[index] * liquidTemp + liq_reg_c[index]), 3)

                    data["obs_time"] = forwardTime
                    data["times"] = times
                    data["type"] = 'mr'
                    mr_bt_dict = data

                    # 正演模拟亮温
                    data = copy.deepcopy(dataTemplate)
                    data["band"] = STANDARD_CHANNELS[number] + "GHz"
                    # 是否有云只判断正演结果时刻
                    data["temp"] = round(forwardBt[index], 3)
                    data["obs_time"] = forwardTime
                    data["times"] = times
                    data["type"] = 'sd'
                    sd_bt_dict = data

                    # 结束匹配
                    break

            if not isMatch:
                print("正演时刻结果未匹配到lv1数据", forwardTime)
                pass
            else:
                """
                剔除数据
                偏差阈值剔除的时候不考虑弱吸收通道，其他通道照常剔除
                35-54.5GHz为弱吸收通道扰动
                """
                # print(float(mr_bt_dict["temp"]), float(sd_bt_dict["temp"]), float(threshold))
                isDelete = False
                # # 剔除条件：非弱吸收通道、阈值不为0、模拟亮温与实测亮温的差值大于所选阈值
                if (float(STANDARD_CHANNELS[number]) < 35 or float(STANDARD_CHANNELS[number]) > 54.5) \
                        and float(threshold) != 0 and abs(
                    float(mr_bt_dict["temp"]) -
                    float(sd_bt_dict["temp"])) > float(threshold):
                    isDelete = True
                    print("剔除通道:", STANDARD_CHANNELS[number], "时刻：", forwardTime)
                    if forwardTime not in removeTime:
                        removeTime.append(forwardTime)
                # 当前通道不剔除
                if not isDelete:
                    btTimesList.append(sd_bt_dict)
                    btTimesList.append(mr_bt_dict)
                    times += 1
                else:
                    # 删除之前通道该时次的数据
                    for btDict in btTimesList[::-1]:
                        if btDict["obs_time"] == forwardTime:
                            btTimesList.remove(btDict)
                            # print("剔除之前通道:", btDict["band"], "时刻：", btDict["obs_time"])

        frequencyList.append(STANDARD_CHANNELS[number])

    for index, number in enumerate(validChannelNumbers):
        for btTime in btTimesList:
            if float(btTime["band"][:6]) == float(
                    STANDARD_CHANNELS[number]) and btTime["type"] == "mr":
                lv1BtList[STANDARD_CHANNELS[number]].append(btTime["temp"])
            if float(btTime["band"][:6]) == float(
                    STANDARD_CHANNELS[number]) and btTime["type"] == "sd":
                forwardBtList[STANDARD_CHANNELS[number]].append(
                    btTime["temp"])

    # 重置times,保证所有通道的时次对齐
    t = 1
    for index in range(0, len(btTimesList), 2):
        if btTimesList[index]["band"] != btTimesList[
            index - 1]["band"] and index > 0:
            t = 1
        btTimesList[index]["times"] = t
        btTimesList[index + 1]["times"] = t
        t += 1

    regParams = []
    dataTemplate = {"band": "", "mr_avg": 0.00, "reg": [], "sd_avg": 0.00}

    # 组织各个频率通道平均值以及拟合曲线参数-----------------------------------------------
    for frequency in frequencyList:
        if len(forwardBtList[frequency]) == 0 or len(
                lv1BtList[frequency]) == 0:
            continue
        data = copy.deepcopy(dataTemplate)
        data["band"] = str(frequency) + "GHz"
        data["mr_avg"] = round(np.mean(lv1BtList[frequency]), 2)
        data["sd_avg"] = round(np.mean(forwardBtList[frequency]), 2)
        data["mr_min"] = round(np.min(lv1BtList[frequency]), 2)
        data["mr_max"] = round(np.max(lv1BtList[frequency]), 2)

        correction = np.polyfit(x=lv1BtList[frequency],
                                y=forwardBtList[frequency],
                                deg=1)
        data["reg"] = [round(correction[0], 2), round(correction[1], 2)]
        regParams.append(data["reg"])

    k, b = [], []
    for ch_params in regParams:
        k.append(ch_params[0])
        b.append(ch_params[1])

    return [{"coeffs": [k, b], "apply": "k*x+b", "condition": ""}]


def main(config, data_source):
    params = {}
    forwardResults = ParseUtils.get_forward_results_by_condition(config, data_source)
    lv1Datas = ParseUtils.get_lv1_by_condition(config, data_source)
    liqParams = liquid_params(config["sounding_station_id"], forwardResults, lv1Datas, threshold=10)
    regParams = reg_params(config["sounding_station_id"], forwardResults, lv1Datas, 10, liqParams)
    params["liqParams"] = liqParams
    params["regParams"] = regParams
    FileUtils.WriteDict2JsonFile(params, r"out/correction",
                                 f"{config['sounding_station_id']}CorrFile{data_source['stime']}to{data_source['etime']}")
    print(params)


if __name__ == '__main__':
    config = {
        "sounding_station_id": "54511",
        "forward_result_path": r"D:/Data/microwave radiometer/Forward Result/sounding",
        "lv1_path": r"D:\Data\microwave radiometer\Measured brightness temperature"
    }

    data_source = {
        "stime": "2021-08-05",
        "etime": "2021-08-19",
    }

    main(config, data_source)
