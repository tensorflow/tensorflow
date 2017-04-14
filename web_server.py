#!/usr/bin/env python
""" This web server will handle the training of the TensorFlow model and the image sets
that will be used for training. """

import json
import os
import re

from flask import (Flask, flash, json, redirect, request, send_from_directory,
                   url_for, jsonify, render_template)
from werkzeug.utils import secure_filename

# Default values
DEFAULTS = {'UploadFolder':'uploads', \
    'AllowedExtensions': 'jpg,jpeg,png,bmp', \
    'StaticURLPath':'static'}

# Paths containing the models
MODELPATHS = {}

# Read server config file
with open('server.config') as f:
    # Read each line and strip all whitespace
    CONTENT = f.readlines()
    CONTENT = [x.strip() for x in CONTENT]

    # insert each line into the DEFAULTS dictionary
    for line in CONTENT:
        entry = line.split('=')
        DEFAULTS[entry[0]] = entry[1]


APP = Flask(__name__, static_url_path=os.path.join('/', DEFAULTS['StaticURLPath']))
APP.config['UPLOAD_FOLDER'] = os.path.join(DEFAULTS['StaticURLPath'], DEFAULTS['UploadFolder'])

# Debug lines
print DEFAULTS
print APP.config['UPLOAD_FOLDER']

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

@APP.route('/images/search', methods=['GET', 'POST'])
def search_images():
    """ Allows the client to search the uploaded images.

    GET: Serves the image search page.
    POST: Receives the plant species to search for and returns a grid of images. """
    if request.method == 'GET':
        # Serve search page
        return APP.send_static_file('search.html')
    if request.method == 'POST':
        # Grab the form data
        exist = False
        search = request.form['species']
        query = search.replace(' ', '_').lower()
        directory = os.path.join(DEFAULTS['UploadFolder'], query)
        full_directory = os.path.join(DEFAULTS['StaticURLPath'], directory)
        results = []

        # Debug lines
        print directory
        print query

        # Check if the directory exists and find all image files if it does
        if os.path.isdir(full_directory):
            for path, subdirs, files in os.walk(full_directory):
                for name in files:
                    # insert the path in the static folder to the image
                    results.append(os.path.join('/', os.path.join(directory, name)))

        # If results list is not empty set exists to true
        if results:
            exist = True

        # Debug lines
        print search
        print exist
        if exist:
            print results

        return render_template('search_results.html', search=search, results=results, exist=exist)

def update_models():
    """ Update the MODELPATHS dictionary with the model paths. """
    # Check if dictionary variables exist
    if 'ModelPath' and 'ModelExtension' and 'LabelExtension' not in DEFAULTS:
        return False
    # Check if path exists
    if not os.path.isdir(DEFAULTS['ModelPath']):
        return False

    # Find all files in the ModelPath with the ModelExtension and LabelExtension
    for path, subdirs, files in os.walk(DEFAULTS['ModelPath']):
        for name in files:
            if name.endswith(DEFAULTS['ModelExtension']):
                file_path = os.path.join(path, name)
                key = file_path \
                    .replace(DEFAULTS['ModelPath'], '') \
                    .replace(DEFAULTS['ModelExtension'], '')

                # Check if the key already exists in the dictionary
                if key in MODELPATHS:
                    MODELPATHS[key]['Model'] = file_path
                else:
                    MODELPATHS[key] = {'Model':file_path}
            if name.endswith(DEFAULTS['LabelExtension']):
                file_path = os.path.join(path, name)
                key = file_path \
                    .replace(DEFAULTS['ModelPath'], '') \
                    .replace(DEFAULTS['LabelExtension'], '')

                # Check if the key already exists in the dictionary
                if key in MODELPATHS:
                    MODELPATHS[key]['Label'] = file_path
                else:
                    MODELPATHS[key] = {'Label':file_path}

        return True

@APP.route('/list-models')
def list_models():
    """ Return the available models in JSON format to the client. """
    if update_models():
        return jsonify(MODELPATHS)
    else:
        return Flask.make_response('error: defined model path does not contain any models', 500)

@APP.route('/update-model')
def download_model():
    """ Send the specified model to the client. """
    # Check that URL arguments have been included.
    if 'model-key' and 'model-time' not in request.args:
        resp = Flask.make_response("error: model-key and/or model-time arguments are missing", 400)
        return resp
    # Update the model locations and check if new version of model exists.
    if update_models():
        key = request.args.get('model-key')
        time = request.args.get('model-time')
        path = MODELPATHS[key]['Model']

        newer = check_time(path, time)

        # Serve model file if newer, else return False.
        if newer:
            path, name = os.path.split(path)
            return send_from_directory(path, name)
        else:
            return False
    else:
        return False

@APP.route('/update-label')
def download_label():
    """ Send the specified label to the client. """
    # Check that URL arguments have been included.
    if 'model-key' and 'model-time' not in request.args:
        resp = Flask.make_response("error: model-key and/or model-time arguments are missing", 400)
        return resp
    # Update the model locations and check if new version of model exists.
    if update_models():
        key = request.args.get('model-key')
        time = request.args.get('model-time')
        path = MODELPATHS[key]['Label']

        newer = check_time(path, time)

        if newer:
            path, name = os.path.split(path)
            return send_from_directory(path, name)
        else:
            return False
    else:
        return False

def check_time(path, time):
    """ Check the metadata of the model file to see if there is a new
    version available. """
    path, name = os.path.split(path)
    server_time = re.findall('\\d+', name)[0]

    if server_time > time:
        return True
    else:
        return False

'''
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

@APP.route('check-version')
def check_version():
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
'''
