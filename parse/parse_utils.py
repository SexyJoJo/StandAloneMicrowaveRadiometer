import copy
import random
from scipy import interpolate
import pandas as pd
from const import TrainConsts
from const.Consts import DEVICE_INFO
import os
from datetime import datetime
from log.log import train_log

# class Mysql:
#     @staticmethod
#     def get_info_by_station(info):
#         conn = sqlalchemy.create_engine('mysql+pymysql://root:123@localhost/microwave?charset=utf8')
#         station_id =
#         config["sounding_station_id"]
#         sql = f"SELECT {info} FROM t_device_info WHERE station_id={station_id}"
#         result = conn.execute(sql).fetchone()[0]
#         if isinstance(result, str):
#             return eval(result)
#         return result

class ParseUtils:
    @staticmethod
    def parse_forward_result(config, result_file):
        """解析正演结果，返回43通道亮温值"""
        brightness_temperature_43channels = []

        data = pd.read_table(
            result_file,
            encoding='gb2312',
            engine='python',
            skiprows=3,
            skipfooter=0,
            sep=r'\s+'
        )

        all_bt = data['BT(K)']
        for i in range(len(all_bt)):
            bt_value = all_bt[i] if all_bt[i] else -999999
            brightness_temperature_43channels.append(bt_value)  # 43通道亮温值

        # 映射
        mapped_bt = []
        mapped_channels = DEVICE_INFO[
            config["sounding_station_id"]]["channels_map"]
        for ch_num in mapped_channels:
            mapped_bt.append(brightness_temperature_43channels[ch_num])
        return mapped_bt

    # @staticmethod
    # def parse_sounding(sounding_file):
    #     with open(sounding_file) as f:
    #         first_line = f.readline()   # 第一行为0层温湿压
    #         first_line = first_line.split()
    #     return first_line[0:3]

    @staticmethod
    def parse_sounding_file(config, obs_time):
        unified_format_file_name = \
            config["sounding_station_id"] + "_" + obs_time + ".txt"
        file_path = os.path.join(
            config["sounding_path"],
            config["sounding_station_id"], obs_time[:4], obs_time[4:6])
        fullPath = os.path.join(file_path,
                                unified_format_file_name)
        # 开始观测时间
        # obsTime = ""
        # if re.match(SOUNDING_UNIFIED_RE_STRING,
        #             sounding_file.unified_format_file_name):
        #     obsTime = sounding_file.unified_format_file_name.split("_")[1][0:14]
        # 文件读取
        try:
            df = pd.read_csv(fullPath, sep=" ", skiprows=0, header=None, engine='python')
            df.iloc[0, 1] += random.randint(-10, 10)
            for col in [0, 1, 2, 3]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            max_height = df[3].max()
            # 对缺测列进行插值
            df.interpolate(inplace=True)

            # 生成线性插值函数
            func_temperature = interpolate.interp1d(df[3],
                                                    df[0],
                                                    fill_value="extrapolate")
            func_pressure = interpolate.interp1d(df[3],
                                                 df[1],
                                                 fill_value="extrapolate")
            func_humidity = interpolate.interp1d(df[3],
                                                 df[2],
                                                 fill_value="extrapolate")

            meteorological_elements_83layers = []
            # 高度层对应的要素值模板
            meteorological_elements_83layer = {
                "height": None,
                "temperature": None,
                "pressure": None,
                "humidity": None,
            }
            # 设备海拔高度,探空数据插值时加上海拔高度
            alt = float(DEVICE_INFO[
                            config["sounding_station_id"]]["alt"])
            height83 = [(i * 1000 + round(alt)) for i in TrainConsts.BASE_HEIGHT83]
            last_pressure = -1
            last_temperature = -1
            for height in height83:
                if height > max_height:
                    pass

                pressure = float(func_pressure(height))
                if pressure == last_pressure:
                    pressure -= 0.001
                last_pressure = pressure

                temperature = 273.16 + func_temperature(height)
                if temperature == last_temperature:
                    temperature -= 0.001
                last_temperature = temperature

                humidity = float(func_humidity(height))
                # 湿度插值异常值处理
                humidity = 0 if humidity < 0 else humidity
                humidity = 100 if humidity > 100 else humidity

                tmp_meteorological_elements_83layer = copy.deepcopy(
                    meteorological_elements_83layer)
                tmp_meteorological_elements_83layer["height"] = height
                tmp_meteorological_elements_83layer["temperature"] = round(
                    temperature, 3)
                tmp_meteorological_elements_83layer["humidity"] = round(
                    humidity, 3)
                tmp_meteorological_elements_83layer["pressure"] = round(
                    pressure, 3)
                meteorological_elements_83layers.append(
                    tmp_meteorological_elements_83layer)
            return meteorological_elements_83layers
        except FileNotFoundError:
            train_log.logger.error(f"未找到{fullPath}探空文件")
            # sys.exit()

    @staticmethod
    def get_forward_results_by_condition(config, data_source):
        """根据设备号和日期筛选所需的正演结果数据"""
        train_log.logger.info("正在解析正演结果")
        results = {}
        selected_path = os.path.join(
            config["forward_result_path"], str(DEVICE_INFO[config["sounding_station_id"]]["id"]),
            str(DEVICE_INFO[config["sounding_station_id"]]["alt"]))
        for root, dirs, files in os.walk(selected_path):
            for filename in files:
                if filename.endswith("IN_liquid_cloud.out"):
                    file_time_str = filename.split(".")[0]
                    file_time = datetime.strptime(file_time_str, "%Y%m%d%H%M%S")
                    stime = datetime.strptime(data_source["stime"], "%Y-%m-%d")
                    etime = datetime.strptime(data_source["etime"], "%Y-%m-%d")
                    if stime <= file_time <= etime:
                        if os.path.getsize(os.path.join(root, filename)) != 0:
                            parse_file = os.path.join(root, filename)
                            results[datetime.strftime(file_time, "%Y%m%d%H%M%S")] = \
                                ParseUtils.parse_forward_result(config, parse_file)
                        else:
                            parse_file = os.path.join(root, file_time_str + '.IN_NOSCALE_IATM1_dn.out')
                            results[datetime.strftime(file_time, "%Y%m%d%H%M%S")] = \
                                ParseUtils.parse_forward_result(config, os.path.join(root, parse_file))
                        print(parse_file)
        train_log.logger.info("解析完毕")
        return results

# print(ParseUtils.get_forward_results_by_condition())
