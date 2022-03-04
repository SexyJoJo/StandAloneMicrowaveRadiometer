import math
from parse.parse_utils import *
from parse.file_utils import FileUtils
from train.train_utils import TrainUtils
from const.Consts import *
from log.log import train_log
import pandas as pd
from const import TrainConsts
from train.cloud_utils import CloudCalculater
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
import random

class Train:
    @staticmethod
    def SaveModelParameters(fileFullName, grid, element, s_scaler, out_scaler):
        """
        保存模型参数
        :fileFullName 模型参数文件所在全路径名
        :grid 最优网络
        """
        clf = grid.best_estimator_
        in_hidden_list = []
        hidden_out_list = []
        submodel_index = 0  # 当前需写入的模型在submodels里的索引值

        # 读已写的模型标准输出文件
        with open(fileFullName, 'r', encoding='utf8') as file:
            file_dict = json.load(file)

        for index, value in enumerate(file_dict["submodels"]):
            if value["elementName"] == '温度' and element == 'temp':
                submodel_index = index
            elif value["elementName"] == '湿度' and element == 'humi':
                submodel_index = index
            elif value["elementName"] == '水汽密度' and element == 'vapor':
                submodel_index = index
            elif value["elementName"] == '液态水含量' and element == 'lwc':
                submodel_index = index

        # 将剩余的数据写入模型标准输出文件json
        now_submodel_dict = file_dict["submodels"][submodel_index]
        now_submodel_dict["nodes"] = [
            len(s_scaler.min_),
            clf.get_params()['hidden_layer_sizes'], clf.n_outputs_
        ]
        # 保存输入、输出归一化参数
        now_submodel_dict["normalization"]["params"] = [[[], []], [[], []]]
        for i in range(len(s_scaler.scale_)):
            now_submodel_dict["normalization"]["params"][0][0].append(
                s_scaler.data_min_[i])

        for i in range(len(s_scaler.scale_)):
            now_submodel_dict["normalization"]["params"][0][1].append(
                s_scaler.data_max_[i])

        for i in range(len(out_scaler.scale_)):
            now_submodel_dict["normalization"]["params"][1][0].append(
                out_scaler.data_min_[i])

        for i in range(len(out_scaler.scale_)):
            now_submodel_dict["normalization"]["params"][1][1].append(
                out_scaler.data_max_[i])

        # 输入层到隐藏层的偏移量, 隐藏层到输出层的偏移量
        now_submodel_dict["biasVectors"] = [[], []]
        now_submodel_dict["biasVectors"][0] = clf.intercepts_[0].tolist()
        now_submodel_dict["biasVectors"][1] = clf.intercepts_[1].tolist()
        for i in range(len(clf.coefs_[0])):
            in_hidden_m = clf.coefs_[0][i].tolist()
            in_hidden_list.append(in_hidden_m)

        for i in range(len(clf.coefs_[1])):
            hidden_out_m = clf.coefs_[1][i].tolist()
            hidden_out_list.append(hidden_out_m)

        # 将生成的权重矩阵list转置成：隐层节点数 * 输入层节点数
        mid_in_pd = pd.DataFrame(in_hidden_list)
        mid_out_pd = pd.DataFrame(hidden_out_list)

        in_hidden_list = mid_in_pd.T.values.tolist()
        hidden_out_list = mid_out_pd.T.values.tolist()

        now_submodel_dict["weightMatrices"] = [[], []]
        now_submodel_dict["weightMatrices"][0] = in_hidden_list
        now_submodel_dict["weightMatrices"][1] = hidden_out_list

        es = clf.__dict__
        now_submodel_dict["estimator"] = {
            'activation': es['activation'],
            'solver': es['solver'],
            'alpha': es['alpha'],
            'batch_size': es['batch_size'],
            'learning_rate': es['learning_rate'],
            'learning_rate_init': es['learning_rate_init'],
            'power_t': es['power_t'],
            'max_iter': es['max_iter'],
            'loss': es['loss'],
            'hidden_layer_sizes': es['hidden_layer_sizes'],
            'shuffle': es['shuffle'],
            'random_state': es['random_state'],
            'tol': es['tol'],
            'verbose': es['verbose'],
            'warm_start': es['warm_start'],
            'momentum': es['momentum'],
            'nesterovs_momentum': es['nesterovs_momentum'],
            'early_stopping': es['early_stopping'],
            'validation_fraction': es['validation_fraction'],
            'beta_1': es['beta_1'],
            'beta_2': es['beta_2'],
            'epsilon': es['epsilon'],
            'n_iter_no_change': es['n_iter_no_change'] if 'n_iter_no_change' in es else None,
            'n_iter_': es['n_iter_'],
            't_': es['t_'],
            'n_layers_': es['n_layers_'],
            'out_activation_': es['out_activation_']
        }

        with open(fileFullName, 'w', encoding='utf8') as parameters_file:
            json.dump(file_dict, parameters_file, ensure_ascii=False, indent=4)

    @staticmethod
    def TrainModel(x_train, y_train, paras_dict):
        """
        训练模型
        :x_train 训练输入
        :y_train 训练输出
        :paras_dict 模型超参数
        """
        # 固定网络初始化的权重参数随机种子
        random.seed(1) 
        np.random.seed(1)

        clf = MLPRegressor(solver='lbfgs',
                           activation='tanh',
                           alpha=1e-7,
                           hidden_layer_sizes=(21,),
                           random_state=0,
                           tol=1e-2,
                           max_iter=200)
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # 某些参数组合将无法收敛，因此此处将其忽略
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    category=ConvergenceWarning,
                                    module="sklearn")
            grid = GridSearchCV(clf,
                                paras_dict,
                                cv=cv,
                                scoring='neg_mean_squared_error',
                                n_jobs=-1)
            grid.fit(x_train, y_train)

        best_clf = grid.best_estimator_
        estimator = best_clf
        print(best_clf)
        print(grid.best_score_)
        print(grid.best_params_)
        print('------')
        return grid

    @staticmethod
    def SampleStandardAndSave(element, taskId, all_input_df, all_output_df,
                              filePath):
        # x_train = []
        # y_train = []
        # for index, value in enumerate(all_input_df):
        #     if index == 0:
        #         x_train = value
        #         y_train = all_output_df[index:index+1]
        #     else:
        #         x_train = x_train.append(value, ignore_index=True)
        #         y_train = y_train.append(all_output_df[index:index+1],
        #                                  ignore_index=True)
        # 空值行滤除
        delete_index = all_input_df[all_input_df.isnull().T.any()].index.tolist()
        print(delete_index)
        all_input_df = all_input_df.drop(index=delete_index)
        all_output_df = all_output_df.drop(index=delete_index)

        x_train = all_input_df
        y_train = all_output_df

        print('x_train')
        print(x_train)
        print('y_train')
        print(y_train)

        if not os.path.exists(filePath):
            os.makedirs(filePath)
        x_train.to_csv(
            os.path.join(filePath, taskId + "_" + element + '_x_train.csv'))

        train_log.logger.info(f"训练样本数目：x:{len(x_train)} y: {len(y_train)}")
        y_train.to_csv(
            os.path.join(filePath, taskId + "_" + element + '_y_train.csv'))
        # minmax标准化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaler.fit_transform(x_train)
        s_scaler = scaler

        scaler_out = MinMaxScaler(feature_range=(-1, 1))
        y_train = scaler_out.fit_transform(y_train)
        out_scaler = scaler_out

        return x_train, y_train, s_scaler, out_scaler

    @staticmethod
    def TrainModelAndSave(config, elements, max_iter, activation, solver,
                          hidden_nodes_range, all_input_df, all_temp_output_df,
                          all_humi_output_df, all_vapor_output_df,
                          modelFullName):
        """
        循环训练各个要素模型
        """
        # 样本标准化以及保存样本文件
        filePath = config["sample_path"]
        # 隐层节点列表
        hidden_list = []
        for i in range(hidden_nodes_range[0], hidden_nodes_range[1] + 1, 1):
            hidden_list.append(i)
        paras_dict = {
            # 'cv_splits': [6, 8, 10],
            # 'test_size': [0.2, 0.3],
            'hidden_layer_sizes': hidden_list,
            # 'max_iter': [100, 150, 200, 250, max_iter],
            'max_iter': [max_iter],
            'activation': [activation],
            'solver': [solver],
            'tol': [0.00001]
            # 'alpha': [0.01]
        }

        for element in elements:
            if element == 'temp':
                x_train, y_train, s_scaler, out_scaler = Train.SampleStandardAndSave(
                    element, '1', all_input_df, all_temp_output_df,
                    filePath)
            if element == 'humi':
                x_train, y_train, s_scaler, out_scaler = Train.SampleStandardAndSave(
                    element, '1', all_input_df, all_humi_output_df,
                    filePath)
            if element == 'vapor':
                x_train, y_train, s_scaler, out_scaler = Train.SampleStandardAndSave(
                    element, '1', all_input_df, all_vapor_output_df,
                    filePath)
            grid = Train.TrainModel(x_train, y_train, paras_dict)
            train_log.logger.info(f"{element}要素模型训练完成")
            # 保存模型参数
            Train.SaveModelParameters(
                modelFullName, grid, element, s_scaler, out_scaler)

    @staticmethod
    def SamplePerturbation(config, input_nodes, inputDF, disrurb, surfaceNodes,
                           cloudNodes):
        k_d = disrurb['k_disturb']
        weak_absorption_d = disrurb['weak_absorption_disturb']
        v_d = disrurb['v_disturb']
        ground_temperature_d = 0.2
        ground_humidity_d = 0.05
        cloud_d = disrurb['cloud_disturb']
        if cloud_d == 0 and k_d == 0 and v_d == 0:
            print('no disturb')
        else:
            # 通道按照频率划分规则
            k_weak_absorption_frequency = 35
            weak_absorption_v_frequency = 54.5
            # 计算出k、弱吸收、v通道的分界点的索引值以及地表温度、湿度的索引值
            k_weak_absorption_index = None
            weak_absorption_v_index = None
            ground_temperature_index = len(input_nodes)
            ground_humidity_index = len(input_nodes) + 1
            # 弱吸收通道扰
            for index, channelNumber in enumerate(input_nodes):
                # standardChannel = StandardChannelDao().getStandardChannelByNumber(channelNumber)
                frequency = float(config["standardChannel"][index])
                if frequency > k_weak_absorption_frequency:
                    k_weak_absorption_index = index
                if frequency > weak_absorption_v_frequency:
                    weak_absorption_v_index = index

            old_df_list = inputDF.values.tolist()
            for i, value in enumerate(old_df_list):
                # 有效节点长度
                length = len(value)
                for j, x in enumerate(value):
                    if j >= ground_temperature_index:
                        # 云层扰动
                        if len(cloudNodes) and j == length - len(
                                cloudNodes) and cloud_d != 0:
                            if x > 0:
                                x_next = old_df_list[i][j + 1]  # 第一层的云厚
                                random_rate = cloud_d * 0.01
                                old_df_list[i][j] = x + np.random.uniform(
                                    -(x * random_rate), x * random_rate)
                                old_df_list[i][j +
                                               1] = x_next + np.random.uniform(
                                    -(x_next * random_rate),
                                    x_next * random_rate)
                                if old_df_list[i][j + 2] != 0 and (
                                        old_df_list[i][j] + old_df_list[i][
                                    j + 1]) > old_df_list[i][j + 2]:
                                    # 差值
                                    diff = (old_df_list[i][j] +
                                            old_df_list[i][j + 1]
                                            ) - old_df_list[i][j + 2]
                                    old_df_list[i][j + 1] = old_df_list[i][
                                                                j + 1] - np.random.uniform(
                                        diff, diff * 2)
                                old_df_list[i][j] = round(old_df_list[i][j])
                                old_df_list[i][j + 1] = round(
                                    old_df_list[i][j + 1])
                            else:
                                break

                        # 温度扰动
                        elif 0 in surfaceNodes and j == ground_temperature_index and ground_temperature_d != 0:
                            old_df_list[i][j] = x + np.random.uniform(
                                -ground_temperature_d, ground_temperature_d)
                        # 湿度扰动
                        elif 1 in surfaceNodes and j == ground_humidity_index and ground_humidity_d != 0:
                            old_df_list[i][j] = x + np.random.uniform(
                                -ground_humidity_d, ground_humidity_d)
                        else:
                            continue

                    # K通道扰动
                    elif k_weak_absorption_index and j < k_weak_absorption_index and k_d != 0:
                        old_df_list[i][j] = x + np.random.uniform(-k_d, k_d)

                    elif weak_absorption_v_index and j < weak_absorption_v_index and weak_absorption_d != 0:
                        old_df_list[i][j] = x + np.random.uniform(
                            -weak_absorption_d, weak_absorption_d)
                    # v通道扰动
                    elif weak_absorption_v_index and j >= weak_absorption_v_index and v_d != 0:
                        old_df_list[i][j] = x + np.random.uniform(-v_d, v_d)

            # 加扰动项之后的df
            disrurb_df = pd.DataFrame(old_df_list,
                                      columns=inputDF.columns.values)

            # 合并之后的df
            inputDF = inputDF.append(disrurb_df, ignore_index=True)

        return inputDF

    @staticmethod
    def InputStandardization(btLists, temp_humi_pres_Lists, cloud_Lists, cloud2_Lists):
        '''
        模型输入类型标准化
        将几个列表矩阵进行组合构成dataframe 标准输入格式
        :btLists 亮温矩阵
        :temp_humi_pres_Lists 温湿压三要素矩阵
        :cloud_Lists 云节点矩阵
        '''
        # 亮温必须参与输入
        input_DF = pd.DataFrame(btLists)
        temp_humi_pres_DF = pd.DataFrame(temp_humi_pres_Lists)
        input_DF = pd.concat([input_DF, temp_humi_pres_DF], axis=1)
        cloud__DF = pd.DataFrame(cloud_Lists)
        input_DF = pd.concat([input_DF, cloud__DF], axis=1)
        cloud2_DF = pd.DataFrame(cloud2_Lists)
        input_DF = pd.concat([input_DF, cloud2_DF], axis=1)
        return input_DF

    @staticmethod
    def OrganizationalColumns(config, selectBtNodeNumbers, surfaceNodes, cloudNodes, cloud2Nodes, isTime):
        """
            动态组织df列头
            selectBtNodeNumbers: 亮温节点索引[0,1,2.....]
            surfaceNodes: 地表温湿压节点索引 [0,1,2]
            cloudNodes: 云节点索引[0,1,2,3,4,5]
            isTime:是否需要时间列
        """

        dateTime = "datetime"
        """
            设备所需模拟亮温所对应的通道频率（用户灵活选择）
            ["22.4","","","",...........]
        """
        frequencyList = []
        standardChannel = config["standardChannel"]
        for channels_number in selectBtNodeNumbers:
            frequencyList.append(standardChannel[channels_number])
        # 温湿压列头
        temp_humi_pres_header = []
        if 0 in surfaceNodes:
            temp_humi_pres_header.append("temperature")
        if 1 in surfaceNodes:
            temp_humi_pres_header.append("humidity")
        if 2 in surfaceNodes:
            temp_humi_pres_header.append("pressure")
        # 云节点列头
        cloud_header = []
        if 0 in cloudNodes:
            cloud_header.append(CLOUD_BOTTOM1)
        if 1 in cloudNodes:
            cloud_header.append(CLOUD_THICK1)
        if 2 in cloudNodes:
            cloud_header.append(CLOUD_BOTTOM2)
        if 3 in cloudNodes:
            cloud_header.append(CLOUD_THICK2)
        if 4 in cloudNodes:
            cloud_header.append(CLOUD_BOTTOM3)
        if 5 in cloudNodes:
            cloud_header.append(CLOUD_THICK3)

        # 云层区间高度节点列头
        cloud2_header = []
        if 0 in cloud2Nodes:
            cloud2_header.append(CLOUD_BOTTOM)
        if 1 in cloud2Nodes:
            cloud2_header.append(CLOUD_TOP)

        header = [dateTime] if isTime else []
        header += frequencyList + temp_humi_pres_header + cloud_header + cloud2_header

        return header

    @staticmethod
    def OrganizeTrainingSamples(config, data_sources, output_nodes, input_nodes):
        # 动态列头
        header = Train.OrganizationalColumns(
            config, input_nodes["btNodes"], input_nodes["surfaceNodes"],
            input_nodes["cloudNodes"], input_nodes["cloud2Nodes"], isTime=False)
        """
        组织训练样本数据
        """

        all_input_df = pd.DataFrame(columns=header)
        all_temp_output_df = []
        all_humi_output_df = []
        all_vapor_output_df = []

        if output_nodes == 83:
            column = [str(i) + "km" for i in TrainConsts.BASE_HEIGHT83]
            # column = Consts.LV2_UNIFIED_HEADER[10:-1]
            all_temp_output_df = pd.DataFrame(columns=column)
            all_humi_output_df = pd.DataFrame(columns=column)
            all_vapor_output_df = pd.DataFrame(columns=column)
        else:
            print("93层输出结果未处理")

        for data_source in list(data_sources):
            # # 查询相应时刻的正演结果模拟亮温数据
            # filters = []
            # filters.append(ForwardResult.wbfsj_id == wbfsj_id)
            # # EC源
            # if data_source["type"]:
            #     filters.append(
            #         ForwardResult.automatic_station_id == data_source["id"])
            # # 探空源
            # else:
            #     filters.append(
            #
            #         ForwardResult.sounding_station_id == data_source["id"])
            # filters.append(ForwardResult.datetime >= data_source["stime"])
            # filters.append(ForwardResult.datetime <= data_source["etime"])
            forwardResults = ParseUtils.get_forward_results_by_condition(config, data_source)
            if not len(forwardResults):
                train_log.logger.warning("所选设备和日期条件无正演结果")
            """
                模拟亮温矩阵
                [
                    [14],
                    [],
                    [],
                    []
                ]
            """
            btLists = []
            # 模拟亮温对应输出标签-高度要素矩阵
            """
                [
                    [83],
                    [],
                    [],
                    []
                ]
            """
            tempLists = []
            humiLists = []
            # 水汽密度
            vaporLists = []
            # 地表温湿压矩阵
            """
                [
                    [3],
                    [],
                    [],
                    []
                ]
            """
            temp_humi_pres_Lists = []
            # 6个云节点矩阵
            """
                [
                    [6],
                    [],
                    [],
                    []
                ]
            """
            cloud_Lists = []

            # 云底云高高度节点矩阵
            """
            [
                [2],
                []
            ]
            """
            cloud2_Lists = []
            """
            扰动参数
            """
            disturb = {
                "cloud_disturb": data_source["cloud_disturb"],
                "k_disturb": data_source["k_disturb"],
                "v_disturb": data_source["v_disturb"],
                "weak_absorption_disturb": data_source["absorb_disturb"]
            }

            for obs_time in forwardResults:
                # btList = []
                tempList = []  # 输出
                humiList = []  # 输出
                vaporList = []  # 输出
                temp_humi_pres_List = []
                cloud_List = []

                btLists.append(forwardResults[obs_time])

                layers83 = ParseUtils.parse_sounding_file(config, obs_time)
                for layer in layers83:
                    layer["temperature"] = 200 if math.isinf(float(layer["temperature"])) else layer["temperature"]
                    tempList.append(layer["temperature"])
                    humiList.append(layer["humidity"])
                    # layer["humidity"] = humiLists[-1] if math.isinf(layer["humidity"]) else layer["humidity"]
                    vapor = TrainUtils.CalculateVapor(layer["temperature"],
                                                      layer["humidity"])
                    vaporList.append(vapor)
                tempLists.append(tempList)
                humiLists.append(humiList)
                vaporLists.append(vaporList)
                # 组织地表温湿压矩阵
                layer0 = layers83[0]

                temp_humi_pres_List = [
                    layer0["temperature"], layer0["humidity"],
                    layer0["pressure"]
                ]
                # 过滤有效温湿压节点
                temp_humi_pres_List = [
                    temp_humi_pres_List[index]
                    for index in range(len(temp_humi_pres_List))
                    if index in input_nodes["surfaceNodes"]
                ]
                temp_humi_pres_Lists.append(temp_humi_pres_List)
                # 组织云数据矩阵
                # cloudData = CloudDataDao().getCloudDataByDateTime(
                #     forwardResult.datetime)
                # 优先使用云雷达数据
                # if cloudData is not None:
                #     cloud_List = [
                #         cloudData.cloud_bottom1,
                #         cloudData.cloud_thickness1,
                #         cloudData.cloud_bottom2,
                #         cloudData.cloud_thickness2,
                #         cloudData.cloud_bottom3,
                #         cloudData.cloud_thickness3,
                #     ]
                # # 根据探空计算云数据
                # else: 暂时没有云数据
                # 探空原数据
                # monortmStandardInputData = MonortmStandardInputDataDao(
                # ).getMonortmStandardInputDataById(
                #     forwardResult.monortm_standard_inputdata_id)
                hei_temp_humi_list = []
                for layer in layers83:
                    if layer["height"] - layers83[0]["height"] > 10000:
                        break
                    hei_temp_humi_list.append([
                        layer["height"], layer["temperature"],
                        layer["humidity"]
                    ])
                # 计算云数据
                cloud_info = CloudCalculater.CalculateCloud(
                    hei_temp_humi_list)
                # 判别组织云数据
                if len(cloud_info) > 0:
                    cloud_List = [
                        cloud_info[0][1],
                        cloud_info[0][3] - cloud_info[0][1]
                    ]
                    if len(cloud_info) > 1:
                        cloud_List += [
                            cloud_info[1][1],
                            cloud_info[1][3] - cloud_info[1][1]
                        ]
                        if len(cloud_info) > 2:
                            cloud_List += [
                                cloud_info[2][1],
                                cloud_info[2][3] - cloud_info[2][1]
                            ]
                        else:
                            cloud_List += [0, 0]
                    else:
                        cloud_List += [0, 0, 0, 0]
                else:
                    # 没有云数据按照晴天来处理
                    cloud_List = [0, 0, 0, 0, 0, 0]

                # 计算最大云区间高度数据
                cloud2_List = CloudCalculater.CalculateCloudIntervalHeight(cloud_List)

                # 过滤有效云节点
                cloud_List = [
                    cloud_List[index] for index in range(len(cloud_List))
                    if index in input_nodes["cloudNodes"]
                ]
                cloud_Lists.append(cloud_List)

                # 过滤云区间高度节点
                cloud2_List = [
                    cloud2_List[index] for index in range(len(cloud2_List))
                    if index in input_nodes["cloud2Nodes"]
                ]
                cloud2_Lists.append(cloud2_List)

            # 输入样本格式化
            input_DF = Train.InputStandardization(
                btLists, temp_humi_pres_Lists, cloud_Lists, cloud2_Lists)

            # 输入加扰动项
            input_DF = Train.SamplePerturbation(
                config, input_nodes["btNodes"], input_DF, disturb,
                input_nodes["surfaceNodes"], input_nodes["cloudNodes"])

            if not input_DF.empty:
                input_DF.columns = header
            # print("加入扰动项之后的样本: ", input_DF)
            temp_DF = pd.DataFrame(tempLists, columns=column)
            humi_DF = pd.DataFrame(humiLists, columns=column)
            vapor_DF = pd.DataFrame(vaporLists, columns=column)

            # 输出不加扰动，样本直接原样翻倍
            temp_DF = temp_DF.append(temp_DF, ignore_index=True)
            humi_DF = humi_DF.append(humi_DF, ignore_index=True)
            vapor_DF = vapor_DF.append(vapor_DF, ignore_index=True)

            all_input_df = all_input_df.append(input_DF, ignore_index=True)

            all_temp_output_df = all_temp_output_df.append(temp_DF)
            all_humi_output_df = all_humi_output_df.append(humi_DF)
            all_vapor_output_df = all_vapor_output_df.append(vapor_DF)
        return all_input_df, all_temp_output_df, all_humi_output_df, all_vapor_output_df

    @staticmethod
    def training(config):
        # try:
        # 输入节点灵活选择
        btNodes = []
        surfaceNodes = []
        cloudNodes = []
        for node in config["input_nodes"]:
            if node == 'surface-all':
                surfaceNodes = [0, 1, 2]
                continue
            if node == 'cloud-all':
                cloudNodes = [0, 1, 2, 3, 4, 5]
                continue
            if node == 'cloud2-interval':
                cloud2Nodes = [0, 1]
                continue
            if node == 'band-all':
                # wbfsj = WBFSJDao().getWBFSJById(wbfsj_id)
                btNodes = DEVICE_INFO[config["sounding_station_id"]]["channels_map"]
                continue
            # 非全部节点
            if "band" in node:
                btNodes.append(int(node.split("-")[-1]))
            if "surface" in node:
                surfaceNodes.append(int(node.split("-")[-1]))
            if "cloud" in node:
                cloudNodes.append(int(node.split("-")[-1]))
            if "cloud2" in node:
                cloud2Nodes.append(int(node.split("-")[-1]))
        input_nodes = {
            "btNodes": btNodes,
            "surfaceNodes": surfaceNodes,
            "cloudNodes": cloudNodes,
            "cloud2Nodes": cloud2Nodes
        }

        train_log.logger.info("正在初始化并保存模型文件...")
        # 初始化并保存模型文件
        modelFullName = TrainUtils.SaveModelParamFile(config, config["data_sources"], config["activation"],
                                                      config["elements"], input_nodes, config["normalization"])
        train_log.logger.info("Done.")

        # 组织输入样本
        train_log.logger.info("正在组织输入样本...")
        # wbfsj_id = CONFIG["wbfsj_id"]
        all_input_df, all_temp_output_df, all_humi_output_df, all_vapor_output_df = Train.OrganizeTrainingSamples(
            config, config["data_sources"], config["output_nodes"], input_nodes)
        train_log.logger.info("Done.")

        if not all_input_df.empty:
            train_log.logger.info("模型训练中...")
            # 训练模型并保存参数
            Train.TrainModelAndSave(
                config, config["elements"], config["max_iter"], config["activation"], config["solver"],
                config["hidden_nodes"], all_input_df, all_temp_output_df,
                all_humi_output_df, all_vapor_output_df, modelFullName)
            # status = 0
            message = "训练成功"
        else:
            # status = 1
            message = "输入样本为0"
        # 训练结束修改训练任务状态
        # trainTask = TrainTaskDao().getTrainTaskByID(trainTask.id)
        # trainTask.status = status
        # TrainTaskDao().updateTrainTask(trainTask)
        # return resultInfo.success(msg=message)
        train_log.logger.info(message)
        return True
        # except Exception as e:
        #     FileUtils.DeleteFile(modelFullName)
        #     train_log.logger.error("fail\n")
        #     print(e)
        #     return False
