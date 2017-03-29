#!/usr/bin/env python
""" This web server will handle the training of the TensorFlow model and the image sets
that will be used for training. """

import os
from stat import *
from flask import Flask, request, redirect, url_for, flash, send_from_directory, json
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'bmp'])

APP = Flask(__name__, static_url_path='/static')
APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """ Check if the attached file contains the valid image extensions defined
    in ALLOWED_EXTENSIONS """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@APP.route('/')
def hello_world():
    """ Index page. TODO: Replace with a dashboard or something to help manage the server """
    return APP.send_static_file('index.html')

@APP.route('/<path:path>')
def static_file(path):
    """ Serve static files inside the static folder. """
    return APP.send_static_file(path)

@APP.route('/images', methods=['GET', 'POST'])
def images():
    """ Handles the training set images

    GET: Displays the image upload page
    POST: Upload the given image to store under the UPLOAD_FOLDER. """
    if request.method == 'GET':
        # Display file upload form
        APP.send_static_file('upload.html')
    if request.method == 'POST':
        # Upload the images attached to the request under the set_name folder
        if 'file' not in request.files:
            flash('No files attached')
            return redirect(request.url)
        img = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without a filename
        if img.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            directory = APP.config['UPLOAD_FOLDER']

            # create the file path if it does not exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            # save the image in the directory
            img.save(os.path.join(directory, filename))
            return redirect('/images')

# Check if model has update
@APP.route('/push-model-update')
def push_model_update():
    """ Send the model file to the client. """
    return send_from_directory('tensorflow/examples/android/assets', \
		'tensorflow_inception_graph.pb')

@APP.route('/push-label-update')
def push_label_update():
    """ Send the lable file to the client. """
    return send_from_directory('tensorflow/examples/android/assets', \
		'imagenet_comp_graph_label_strings.txt')

@APP.route('/check-version')
def check_version():
    """ Check the metadata of the model file to see if there is a new
    version available. """
    update_available = False
    client_mod_time = int(request.args.get('time-modified'))
    client_size = int(request.args.get('size'))

    file_info = os.stat('tensorflow/examples/android/assets/tensorflow_inception_graph.pb')
    size = file_info[ST_SIZE]
    mod_time = file_info[ST_MTIME]

    print 'client mod time: {0}, client size: {1}'.format(client_mod_time, client_size)
    print 'server mod time: {0}, server size: {1}'.format(mod_time, size)

    # compare the last modified time first
    if client_mod_time < mod_time:
        print 'client mod time is older than server mod time'
        update_available = True
        return APP.response_class(
            response=json.dumps(update_available),
            status=200,
            mimetype='application/json'
        )

    if client_size != size:
        print 'client size is not the same as server size'
        update_available = True
        return APP.response_class(
            response=json.dumps(update_available),
            status=200,
            mimetype='application/json'
        )

    print 'no update available'
    return APP.response_class(
        response=json.dumps(update_available),
        status=200,
        mimetype='application/json'
    )
