from trash_detection.src.utils import create_model, make_detection
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__, template_folder = 'template')

faster_rcnn_mobilenet_v2 = create_model('plastic_bottle_detection_faster_rcnn_mobilenetnetv2.pth')
faster_rcnn_resnet50_v1 = create_model('plastic_bottle_detection_faster_rcnn_resnet50v1.pth')
models = {'faster_rcnn_mobilenet_v2': faster_rcnn_mobilenet_v2,
          'faster_rcnn_resnet50_v1': faster_rcnn_resnet50_v1}


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/detection', methods = ["POST"])
def detect_trash():
    if request.method == 'POST':
        model_type = request.form['model_type']
        image_file = request.files['image-file']
        im = make_detection(models[model_type], image_file)

        #TODO: Set up the Results Page


