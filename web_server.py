#!/usr/bin/env python
""" This web server will handle the training of the TensorFlow model and the image sets
that will be used for training. """

import json
import os
import re
import math

from flask import (Flask, flash, json, redirect, request, send_from_directory,
                   url_for, jsonify, render_template, make_response)
from werkzeug.utils import secure_filename

# Default values
DEFAULTS = { \
    'StaticURLPath':'static', \
    'DatabaseFolder':'database', \
    'UploadFolder':'uploads', \
    'AllowedExtensions':'jpg,jpeg,png,bmp', \
    'ModelPath':'tensorflow/examples/android/assets', \
    'ModelExtension':'.pb', \
    'LabelExtension':'.txt' \
}

# Check if server config file exists
# If if doesn't, create the file
if not os.path.isfile('server.config'):
    conf = open('server.config', 'w+')
    for key, value in DEFAULTS.iteritems():
        conf.write(key + '=' + value + '\n')

    conf.close()

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

def allowed_file(filename):
    """ Check if the attached file contains the valid image extensions defined
    in DEFAULTS['AllowedExtensions'] """
    allowed_extensions = set(DEFAULTS['AllowedExtensions'].split(','))
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions

@APP.route('/')
def index():
    """ Index page. TODO: Replace with a dashboard or something to help manage the server """
    return render_template("index.html")

@APP.route('/<path:path>')
def static_file(path):
    """ Serve static files inside the static folder. """
    return APP.send_static_file(path)

@APP.route('/' + DEFAULTS['RootTFFiles'] + '/<path:path>')
def serve_db_image(path):
    """ Serve the images inside the database folder. """
    path = os.path.join(DEFAULTS['RootTFFiles'], path)
    path, name = os.path.split(path)
    return send_from_directory(path, name)

# TODO: Change the image upload and search process to use the correct file
# structure
@APP.route('/images', methods=['GET', 'POST'])
def images():
    """ Handles the training set images

    GET: Displays the image upload page
    POST: Upload the given image to store under the UPLOAD_FOLDER. """
    if request.method == 'GET':
        # Display file upload form
        return render_template('upload.html')
    if request.method == 'POST':
        # Check if the form data was completed successfully
        if 'species' not in request.form:
            return make_response('error: species field not specified', 400)
        if 'file' not in request.files:
            return make_response('error: no file(s) uploaded', 400)

        # Grab the form data
        species = request.form['species'].replace(' ', '_').lower()
        imgs = request.files.getlist('file')

        # Check if no file was selected
        if not imgs:
            return make_response('error: no file(s) selected', 400)

        # Create the file path if it does not exist
        directory = os.path.join( \
            DEFAULTS['DatabaseFolder'], \
            species.replace(' ', '_'))
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Process file if all information is present
        for img in imgs:
            if img and allowed_file(img.filename) and species:
                # Count the number of files in the species directory to figure out
                # what number to assign to the uploaded image
                number = len([name for name in os.listdir(directory) if os.path.isfile(name)])
                extension = img.filename.rsplit('.', 1)[1].lower()
                filename = species.replace(' ', '_') + '_' + str(number).zfill(4) + '.' + extension

                # Check if file name already exists
                while os.path.isfile(os.path.join(directory, filename)):
                    number = number + 1
                    filename = species.replace(' ', '_') + '_' + str(number).zfill(4) + '.' + extension

                # save the image in the directory
                img.save(os.path.join(directory, filename))
                print "File saved successfully:" + filename


        return make_response('file(s) uploaded successfully', 200)

def grab_images(search, number, page):
    # initialize variables
    exist = False
    query = search.replace(' ', '_').lower()
    directory = DEFAULTS['DatabaseFolder']
    results = []

    # Check if the directory exists and find all image files if it does
    if os.path.isdir(directory):
        for path, subdirs, files in os.walk(directory):
            for subdir in subdirs:
                if query in subdir:
                    for filepath, subsubdirs, subfiles in os.walk(os.path.join(path, subdir)):
                        for name in subfiles:
                            if allowed_file(name):
                                results.append('/' + os.path.join(filepath, name))

    # If results list is not empty set exists to true
    if results:
        exist = True

    # Return all images if number or page are -1
    if number == -1 or page == -1:
        return render_template('search_results.html', search=search.capitalize(), results=results, exist=exist)
    # Return paginated result if number and page are not -1
    else:
        # Calculate pagination info
        page_start = (page-1) * number
        page_end = page_start + number
        pages = int(math.ceil(float(len(results)) / number))

        # Check if pageStart is within the results list
        if page_start >= 0 and page_start < len(results) - 1:
            results_page = []

            # Check if pageEnd is within the results list end
            if page_end <= len(results):
                results_page = results[page_start:page_end]
            else:
                results_page = results[page_start:]

            return render_template('search_results.html', search=search.capitalize(), results=results_page, exist=exist, paginated=True, entries=number, page=page, pages=pages)
        else:
            return render_template('search_results.html', search=search.capitalize(), results=[], exist=False, paginated=True, entries=number, page=1, pages=pages)


@APP.route('/images/search', methods=['GET', 'POST'])
def search_images():
    """ Allows the client to search the uploaded images.

    GET: Serves the image search page.
    POST: Receives the plant species to search for and returns a grid of images. """
    if request.method == 'GET':
        # Check if arguments are present
        if 'species' in request.args:
            species = request.args.get('species')

            # Return paginated images
            if 'entries' and 'page' in request.args:
                entries = int(request.args.get('entries'))
                page = int(request.args.get('page'))

                return grab_images(species, entries, page)

            # Return all images
            return grab_images(species, -1, -1)

        # Serve search page
        return render_template('search.html')
    if request.method == 'POST':
        # Check if form data was completed successfully
        if 'species' not in request.form:
            return make_response('error: species field not specified', 400)

        return grab_images(request.form['species'], -1, -1)

@APP.route('/update-model')
def download_model():
    """ Send the specified model to the client. """
    # Check that URL arguments have been included.
    if 'model-key' not in request.args:
        return make_response('error: model-key argument missing from URL', 400)
    #if 'model-time' not in request.args:
    #    return make_response('error: model-time argument missing from URL', 400)

    key = request.args.get('model-key')
    time = request.args.get('model-time')
    modelPath = DEFAULTS[key]

    for path, subdirs, files in os.walk(modelPath):
        for name in files:
            if name.endswith(DEFAULTS['ModelExtension']):
                path = os.path.join(path, name)
                newer = check_time(path, time)

                # Serve model file if newer, else return False.
                if newer:
                    path, name = os.path.split(path)
                    return send_from_directory(path, name)
    return make_response('error: model file not found', 404)

@APP.route('/update-label')
def download_label():
    """ Send the specified label to the client. """
    # Check that URL arguments have been included.
    if 'model-key' not in request.args:
        return make_response('error: model-key argument missing from URL', 400)
    if 'model-time' not in request.args:
        return make_response('error: model-time argument missing from URL', 400)

    key = request.args.get('model-key')
    time = request.args.get('model-time')
    modelPath = DEFAULTS[key]

    for path, subdirs, files in os.walk(modelPath):
        for name in files:
            if name.endswith(DEFAULTS['LabelExtension']):
                if 'bottlenecks' not in path:
                    newer = check_time(name, time)

                    # Serve model file if newer, else return False.
                    if newer:
                        return send_from_directory(path, name, as_attachment=True)
    return make_response('error: model file not found', 404)

def check_time(path, time):
    """ Check the metadata of the model file to see if there is a new
    version available. """
    path, name = os.path.split(path)
    digits = re.findall('\\d+', name)
    if digits:
        server_time = re.findall('\\d+', name)[-1]

        if server_time > time:
            return True
        else:
            return False
    else:
        return False

@APP.route('/download-apk')
def download_apk():
    apk_path = DEFAULTS['APKPath']

    for path, subdirs, files in os.walk(apk_path):
        for name in files:
            if name.endswith('.apk'):
                return send_from_directory(path, name, as_attachment=True)

    return make_response('error: apk file not found', 404)
