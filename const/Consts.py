import json

with open(r"config/device_info.json", "r", encoding='utf8') as f:
    DEVICE_INFO = json.load(f)
with open(r"config/liquid_channel.json", "r", encoding='utf8') as f:
    LIQUID_INFO = json.load(f)

# 云文件数据表头常亮
CLOUD_TIME = "Time"
CLOUD_BOTTOM1 = "Bottom1"
CLOUD_TOP1 = "Top1"
CLOUD_THICK1 = "Thick1"
CLOUD_BOTTOM2 = "Bottom2"
CLOUD_TOP2 = "Top2"
CLOUD_THICK2 = "Thick2"
CLOUD_BOTTOM3 = "Bottom3"
CLOUD_TOP3 = "Top3"
CLOUD_THICK3 = "Thick3"

# 云层区间高度文件数据表头常量
CLOUD_BOTTOM = "Bottom"
CLOUD_TOP = "Top"

# 探空统一格式文件名匹配正则：
# SOUNDING_UNIFIED_RE_STRING = "^[\d]{5}[\_][\d]{14}.[Tt][Xx][Tt]$"

STANDARD_CHANNELS = ["22.234", "22.235", "22.240", "22.500", "23.034", "23.035", "23.040", "23.834", "23.835", "23.840", "25.000", "25.440",
                     "26.234", "26.235", "26.240", "27.840", "28.000", "30.000", "31.400", "51.248", "51.250", "51.260", "51.760", "52.280",
                     "52.800", "52.804", "53.336", "53.340", "53.848", "53.850", "53.860", "54.400", "54.940", "55.500", "56.020", "56.660",
                     "57.288", "57.290", "57.300", "57.960", "57.964", "58.000", "58.800"]