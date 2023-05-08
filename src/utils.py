import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype
from torchvision.transforms.functional import to_pil_image

def create_model(file_path):
    model = fasterrcnn_mobilenet_v3_large_fpn(torch.load(file_path))
    
    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def make_detection(model, image_file):

    with torch.inference_mode():
        img = read_image(image_file)

        preprocess = Compose([
            ToTensor(),
            ConvertImageDtype(dtype = torch.float32)
        ])

        batch = [preprocess(img)]

        prediction = model(batch)[0]
        class_to_label = {0: 'background', 1: 'plastic bottle'}
        labels = [class_to_label[i] for i in prediction["labels"]]

        box = draw_bounding_boxes(img,
                                    boxes = prediction['boxes'],
                                    labels = labels,
                                    colors = 'green',
                                    width = 4,
                                    font_size = 30)

        im = to_pil_image(box.detach())

        return im



