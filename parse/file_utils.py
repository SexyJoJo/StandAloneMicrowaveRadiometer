import json
import os


class FileUtils:
    @staticmethod
    def WriteDict2JsonFile(dict, basePath, fileName):
        """
            将dict写到指定指定文件
                dict: 数据字典
                basePath：基础路径
                fileName: 文件名
        """
        fileName = fileName + ".json"
        save_path = os.path.join(basePath, fileName)
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        with open(save_path, 'w', encoding="utf8") as file:
            json.dump(dict, file, ensure_ascii=False, indent=4)
        return save_path

    @staticmethod
    def DeleteFile(filePath):
        if os.path.exists(filePath):
            print("delete %s" % filePath)
            os.remove(filePath)
        else:
            print("no this file")