#!/usr/bin/env python
""" This web server will handle the training of the TensorFlow model and the image sets
that will be used for training. """

import os
from stat import *

from flask import (Flask, flash, json, redirect, request, send_from_directory,
                   url_for)
from werkzeug.utils import secure_filename

# Default values
DEFAULTS = {'UploadFolder':'./static/uploads', \
    'AllowedExtensions': 'jpg,jpeg,png,bmp', \
    'StaticURLPath':'/static'}

# Read server config file
with open('server.config') as f:
    # Read each line and strip all whitespace
    CONTENT = f.readlines()
    CONTENT = [x.strip() for x in CONTENT]

    # insert each line into the DEFAULTS dictionary
    for line in CONTENT:
        entry = line.split('=')
        DEFAULTS[entry[0]] = entry[1]


APP = Flask(__name__, static_url_path=DEFAULTS['StaticURLPath'])
APP.config['UPLOAD_FOLDER'] = DEFAULTS['UploadFolder']

def allowed_file(filename):
    """ Check if the attached file contains the valid image extensions defined
    in DEFAULTS['AllowedExtensions'] """
    allowed_extensions = set(DEFAULTS['AllowedExtensions'].split(','))
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions

@APP.route('/')
def index():
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
        return APP.send_static_file('upload.html')
    if request.method == 'POST':
        # Check if the form data was completed successfully
        if 'species' not in request.form:
            flash('Plant species not specified')
            return redirect(request.url)
        if 'file' not in request.files:
            flash('No files attached')
            return redirect(request.url)

        # Grab the form data
        species = request.form['species'].replace(' ', '_').lower()
        img = request.files['file']
        extension = img.filename.rsplit('.', 1)[1].lower()

        # Debug lines
        print 'species: {0}'.format(species)

        # Check if no file was selected
        if img.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Process file if all information is present
        if img and allowed_file(img.filename) and species:
            # Create the file path if it does not exist
            directory = os.path.join(APP.config['UPLOAD_FOLDER'], species.replace(' ', '_'))
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Count the number of files in the species directory to figure out
            # what number to assign to the uploaded image
            number = len([name for name in os.listdir(directory) if os.path.isfile(name)])
            filename = species.replace(' ', '_') + '_' + str(number).zfill(4) + '.' + extension

            # Check if file name already exists
            while os.path.isfile(os.path.join(directory, filename)):
                number = number + 1
                filename = species.replace(' ', '_') + '_' + str(number).zfill(4) + '.' + extension

                # Debug lines
                print 'number incremented to {0}'.format(number)

            # Debug lines
            print 'directory: {0}'.format(directory)
            print 'file name: {0}'.format(filename)

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
    # Grab the URL arguments
    update_available = False
    client_mod_time = int(request.args.get('time-modified'))
    client_size = int(request.args.get('size'))

    # Grab the metadata for the model file in the server
    file_info = os.stat('tensorflow/examples/android/assets/tensorflow_inception_graph.pb')
    size = file_info[ST_SIZE]
    mod_time = file_info[ST_MTIME]

    # Debug lines
    print 'client mod time: {0}, client size: {1}'.format(client_mod_time, client_size)
    print 'server mod time: {0}, server size: {1}'.format(mod_time, size)

    # Compare the last modified time first
    if client_mod_time < mod_time:
        print 'client mod time is older than server mod time'
        update_available = True
        return APP.response_class(
            response=json.dumps(update_available),
            status=200,
            mimetype='application/json'
        )

    # Compare size if modified time is the same (which should not happen)
    if client_size != size:
        print 'client size is not the same as server size'
        update_available = True
        return APP.response_class(
            response=json.dumps(update_available),
            status=200,
            mimetype='application/json'
        )

    # No update available, return false
    print 'no update available'
    return APP.response_class(
        response=json.dumps(update_available),
        status=200,
        mimetype='application/json'
    )
