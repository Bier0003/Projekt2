import os 
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
import tensorflow as tf
import numpy as np




app = Flask(__name__)

#uplaod folder and allowed file types
UPLOAD_BASE_DIR = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def predict_shose_label(image_storage_object):
    """
    upload picture from page -> Predict the type of shoe in the given image -> return the predicted shoe type as a string
    """
   #upload and resize the image for AI model 
    img = Image.open(image_storage_object)
    img_resized = img.resize((224, 224))
    img_color = img_resized.convert('RGB')

    #process the image for AI model
    img_array = np.array(img_color)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # AI prdiction
    predictions = model.predict(img_array)
    results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

    #pull the predicted shoe type from the results
    predicted_label = results[0][1]
    return predicted_label


#flask routes
@app.route('/', methods=['GET'])
def index_page():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_BASE_DIR, filename)

@app.route('/upload',methods=['POST'])
def handle_upload():
    if 'shoe_properties' not in request.files:
        return " no file part in the request ", 400
    
    file = request.files ['shoe_properties']
    
    if file.filename == '':
        return "no file selected for uploading", 400
    
    if file:
        detected_label = predict_shose_label(file)
        folder_name = detected_label.replace(" ", "_")
        target_folder = os.path.join(UPLOAD_BASE_DIR, folder_name)

        os.makedirs(target_folder, exist_ok=True)
        save_path = os.path.join(target_folder, file.filename)
        file.seek(0)  # Reset file pointer to the beginning after saving
        file.save(save_path)

    return render_template('success.html', detected_label=detected_label, saved_path=save_path)

#open port for flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)


