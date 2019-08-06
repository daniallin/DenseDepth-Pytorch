from models.backbone.DenseDepth import DenseDepthModel


def build_model(model_name):
    if model_name == 'DenseDepth':
        return DenseDepthModel()
    else:
        raise NotImplementedError
