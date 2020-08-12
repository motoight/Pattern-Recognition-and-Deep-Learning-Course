import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def construct_models():

    # set pretrained false to use us weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False,pretrained_backbone=False)
    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
