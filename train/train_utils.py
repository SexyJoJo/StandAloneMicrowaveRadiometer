from datetime import datetime

import parse
from parse.file_utils import FileUtils
from const.Consts import CONFIG, DEVICE_INFO
from math import log10


class TrainUtils:
    @staticmethod
    def CalculateVapor(temp, humi):
        T0 = 273.16
        T = temp
        U = humi
        logEw = 10.79574 * (1 - T0 / T) - 5.028 * log10(T / T0) + \
                1.50475 * 0.0001 * (1 - 10 ** (-8.2969 * (T / T0 - 1))) + \
                0.42873 * 0.001 * (10 ** (4.76955 * (1 - T0 / T)) - 1) + 0.78614
        e = (10 ** logEw) * (U / 100)
        vapor = 216.7679 * e / T
        return vapor

    @staticmethod
    def SaveModelParamFile(data_sources, activation,
                           elements, input_nodes, normalization):
        model_json = {
            "name": CONFIG["model_name"],
            "ctime": datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
            "compatible": True,
            "private": False,
            "trainingData": []
        }
        # model_json["name"] = trainTask.name
        # model_json["description"] = trainTask.description
        # user = UserDao().getUserById(trainTask.user_id)
        # model_json["author"] = {
        #     "name": user.username,
        # }
        #
        # wbfsj = WBFSJDao().getWBFSJById(wbfsj_id)
        # wbfsjModel = WBFSJModelDao().getWBFSJModelById(wbfsj.wbfsjmodel_id)
        # manufacturer = ManufacturerDao().getManufacturerById(
        #     wbfsjModel.manufacturer_id)
        # model_json["equipment"] = {
        #     "name": wbfsj.name,
        #     "brand": manufacturer.name,
        #     "type": wbfsjModel.name,
        #     "bands": Gson.JsonStr2List(wbfsj.channels_number),
        #     # "bands": [int(i) for i in wbfsj.channels_number.split(',')],
        # }
        disturb = []

        for source in data_sources:
            # id = source["id"]
            stime = source["stime"]
            etime = source["etime"]
            disturb.append({
                'cloud_disturb': source["cloud_disturb"],
                'k_disturb': source["k_disturb"],
                'v_disturb': source["v_disturb"],
                'weak_absorption_disturb': source["absorb_disturb"]
            })
            # soundingStation = SoundingStationDao().getSoundingStationByID(
            #     id)
            # automaticStation = AutomaticStationDao().getAutomaticSationByID(
            #     id)

            one_dict = {"station": CONFIG["wbfsj_id"], "interval": [stime, etime]}
            # one_dict["dataType"] = "EC数据" if source["type"] else "探空数据"
            model_json["trainingData"].append(one_dict)
        # 默认没有该字段key，在反演时就不进行偏差订正
        # model_json["regParams"] = [{
        #     "coeffs": [[], []],
        #     "apply": "k*x+b",
        #     "condition": ''
        # }]

        # model_json["liqParams"] = [{
        #     "coeffs": [[], []],
        #     "apply": "k*x+b",
        #     "condition": ''
        # }]

        model_json["elements"] = []
        model_json["submodels"] = []

        for element in elements:
            if element == 'temp':
                model_json["elements"].append('温度')
            if element == 'vapor':
                model_json["elements"].append('水汽密度')
            if element == 'humi':
                model_json["elements"].append('湿度')
            if element == 'lwc':
                model_json["elements"].append('液态水含量')

        for element in model_json["elements"]:
            one_dict = {}
            if element == '温度':
                one_dict["elementName"] = '温度'
                one_dict["elementId"] = 11
            elif element == '水汽密度':
                one_dict["elementName"] = '水汽密度'
                one_dict["elementId"] = 12
            elif element == '湿度':
                one_dict["elementName"] = '湿度'
                one_dict["elementId"] = 13
            elif element == '液态水含量':
                one_dict["elementName"] = '液态水含量'
                one_dict["elementId"] = 14

            one_dict["condition"] = ''
            one_dict["activation"] = activation
            one_dict["weightMatrices"] = []
            one_dict["normalization"] = {
                "enabled": [True, True],
                "methods": [normalization, normalization],
                "params": [],
            }
            one_dict["biasVectors"] = []
            one_dict["nodes"] = []

            model_json["submodels"].append(one_dict)

        # 保存输入节点的通道索引等信息
        # 构建偏差订正、液态水订正参数数组和实际模型数组通道的映射关系
        origin_equip_bands = DEVICE_INFO[CONFIG["wbfsj_id"]]["channels_map"]
        # print(origin_equip_bands)
        mapping_bands = []  # 反演程序用
        for i, v in enumerate(origin_equip_bands):
            if v in input_nodes["btNodes"]:
                mapping_bands.append(i)
        # print(mapping_bands)
        model_json["input_nodes"] = {
            'inputBtNodes': input_nodes["btNodes"],
            'surfaceNodes': input_nodes["surfaceNodes"],
            'cloudNodes': input_nodes["cloudNodes"],
            'mappingNodes': mapping_bands
        }

        # 保存初始化的模型数据字典
        modelFullName = FileUtils.WriteDict2JsonFile(model_json,
                                                     CONFIG["model_path"],
                                                     CONFIG["model_name"])

        return modelFullName
