#!/usr/bin/env python
""" This web server will handle the training of the TensorFlow model and the image sets
that will be used for training. """

import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'user_images'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'bmp'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return 'Hello world!'

# Training API
#@app.route('/images', methods=['GET'])
#def images():
#    if request.method == 'GET':
#        

@app.route('/images', defaults={'set_name': None})
@app.route('/images?set_name=<set_name>', methods=['GET', 'POST'])
def image_upload(set_name):
    if request.method == 'GET':
        # Display all the images stored under the set_name folder
        if set_name != None:
            return 'Display images from %s' % set_name
        else:
            return '''
            <!doctype html>
            <title>Upload new image</title>
            <h1>Upload new image</h1>
            <form method=post enctype=multipart/form-data>
                <p><input type=file name=file>
                <input type=text name=set_name value=default>
                <input type=submit value=Upload></p>
            </form>
            '''
    if request.method == 'POST':
        # Upload the images attached to the request under the set_name folder
        if 'file' not in request.files:
            flash('No files attached')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without a filename
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], set_name, filename))
            return redirect(url_for('uploaded_file', filename=filename))