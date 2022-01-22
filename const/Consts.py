import json

with open(r"config/device_info.json", "r", encoding='utf8') as f:
    DEVICE_INFO = json.load(f)

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

# 探空统一格式文件名匹配正则：
# SOUNDING_UNIFIED_RE_STRING = "^[\d]{5}[\_][\d]{14}.[Tt][Xx][Tt]$"
