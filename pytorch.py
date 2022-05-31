import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("models/pointrend_resnet50.pkl")
ins.segmentImage("input_images/eliott-reyna-jCEpN62oWL4-unsplash.jpg", show_bboxes=True, 
output_image_name="output_images/output-pytorch-01.jpg")
ins.segmentImage("input_images/pexels-sebastian-voortman-214576.jpg", show_bboxes=True, 
output_image_name="output_images/output-pytorch-02.jpg")
