import detectron2.config as dconfig

def default_config(config_path, weights, device: str = "cpu"):
    """
    Add config for VIT.
    """
    config = dconfig.get_cfg()
    config.MODEL.VIT = dconfig.CfgNode()

    # CoaT model name.
    config.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    config.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]
    config.MODEL.VIT.IMG_SIZE = [224, 224]
    config.MODEL.VIT.POS_TYPE = "shared_rel"
    config.MODEL.VIT.DROP_PATH = 0.
    config.MODEL.VIT.MODEL_KWARGS = "{}"
    config.SOLVER.OPTIMIZER = "ADAMW"
    config.SOLVER.BACKBONE_MULTIPLIER = 1.0
    config.AUG = dconfig.CfgNode()
    config.AUG.DETR = False

    config.merge_from_file(config_path)
    config.MODEL.WEIGHTS = weights
    config.MODEL.DEVICE = device

    return config


def add_vit_config(cfg):
    """
    Add config for VIT.
    """
    _C = cfg

    _C.MODEL.VIT = dconfig.CfgNode()

    # CoaT model name.
    _C.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]

    _C.MODEL.VIT.IMG_SIZE = [224, 224]

    _C.MODEL.VIT.POS_TYPE = "shared_rel"

    _C.MODEL.VIT.DROP_PATH = 0.

    _C.MODEL.VIT.MODEL_KWARGS = "{}"

    _C.SOLVER.OPTIMIZER = "ADAMW"

    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0

    _C.AUG = dconfig.CfgNode()

    _C.AUG.DETR = False