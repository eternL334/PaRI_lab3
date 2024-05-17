from flask import Flask
from flask import render_template, redirect
from flask import request, url_for
from flask import send_file
from waitress import serve
from werkzeug.utils import secure_filename
import os   
import cv2 as cv
import numpy as np
from model import segmentate, remove_components, get_vertices, get_graph, squeeze_sizes, get_class
from skimage.morphology import skeletonize

app = Flask(__name__)
UPLOAD_FOLDER = 'src/static'
text = ''

def make_prediction(low, high, min_size, radius):
    global text

    image = cv.imread(os.path.join(UPLOAD_FOLDER, "image.png"))
    mask = segmentate(image, low, high)
    cv.imwrite(os.path.join(UPLOAD_FOLDER, "segmentate.png"), mask)

    skeleton = skeletonize(mask).astype(np.uint8)
    skeleton = remove_components(skeleton, min_size)
    vertices = get_vertices(skeleton)

    sizes = squeeze_sizes(get_graph(vertices, radius))
    idx = get_class(sizes)

    text = f'Sizes: {sizes}\nClass: {idx + 1}'.split('\n')

    cv.imwrite(os.path.join(UPLOAD_FOLDER, "skeleton.png"), skeleton * 255)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def index_post():
    try:
        file = request.files['image']
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        image = cv.imread(os.path.join(UPLOAD_FOLDER, filename))
        os.remove(os.path.join(UPLOAD_FOLDER, filename))
        cv.imwrite(os.path.join(UPLOAD_FOLDER, "image.png"), image)

        low = np.array([
            request.form.get("lowhue"),
            request.form.get("lowsaturation"),
            request.form.get("lowvalue"),
        ], dtype=np.uint8)

        high = np.array([
            request.form.get("highhue"),
            request.form.get("highsaturation"),
            request.form.get("highvalue"),
        ], dtype=np.uint8)

        min_size = int(request.form.get("size"))
        radius = int(request.form.get("radius"))

        make_prediction(low, high, min_size, radius)

        return redirect(url_for('predict'))
    except Exception as e:
        app.logger.warning(f"{e}")
        return redirect(url_for('fail'))

@app.route("/predict")
def predict():
    return render_template("predict.html", text=text)

@app.route("/fail")
def fail():
    return render_template("fail.html")

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port='5000')
