from models.DenseDepth import DenseDepthModel
from models.resdeep.ResDeep import ResDeep


def build_model(model_name):
    if model_name == 'DenseDepth':
        return DenseDepthModel()
    elif model_name == 'ResDeep':
        return ResDeep(output_scale=16)
    else:
        raise NotImplementedError


