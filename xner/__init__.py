from enum import Enum, unique


# TODO: 以后尝试引用枚举，现在暂时使用string
@unique
class LabelType(Enum):
    biso = "biso"
    bmeso = "bmeso"


SETTINGS = {
    'labels': ["PRV", "CTY", "CNTY", "TWN", "CMNT", "RD", "NO", "POI", "O"],
    'label_type': "bmeso",  # another is 'biso'
    'crf_features': ["BasicFeature", "NgramFeature", "DistrictFeature"]
}


def set_option(attr, value):
    global SETTINGS
    if attr == "labels":
        value = [v.upper() for v in value]
    SETTINGS[attr] = value
