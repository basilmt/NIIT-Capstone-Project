from flask import Flask, redirect, flash, render_template, request
import os
import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image as im

app = Flask(__name__)

model=None

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

ALLOWED_EXTENSIONS = {'dcm'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods = ['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template("index.html")
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Only dcm files allowed')
            return redirect(request.url)
        
        if file:
            filename = 'img.dcm'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return render_template("result.html")
            predict = predict_from_file_name(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if predict > 0.5:
                predict = f"{100 * predict:.2f}% Chance to have pneumonia. Please consult a doctor to validate this result"
            else:
                predict = "It was observed that you may not have pneumonia"
            
            return render_template("result.html", conclusion=predict)

        return redirect(request.url)


def load_img_into_ndarray(file_name):
    data_row_img_data = pydicom.read_file(file_name)
    ar = data_row_img_data.pixel_array
    data = im.fromarray(ar)
    data.save("static\img.png")
    ar = ar[..., np.newaxis] 
    return ar/255.0


def predict_from_file_name(file_name):
    l = []
    l.append(load_img_into_ndarray(file_name))
    l = np.array(l)
    # model = getmodel()
    return model.predict(l)[0][0]


if __name__ == '__main__':
    model = tf.keras.models.load_model('saved_model\my_model2_78acc')
    app.run()
