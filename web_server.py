#!/usr/bin/env python
""" This web server will handle the training of the TensorFlow model and the image sets
that will be used for training. """

import os, time
from os import listdir
from os.path import isfile, join
from stat import *
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './user_images'
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
	return APP.send_static_file(path)
	
# Training API
# Receive images and upload them to the set folder
@APP.route('/images', methods=['GET', 'POST'])
def images():
    """ Handles the training set images

    GET: Displays the image upload page if no set_name argument is specified
    in the URL or displays all the images of a given set if set_name is specified
    in the URL

    POST: Upload the given image to store under the UPLOAD_FOLDER/set_name, where
    set_name is specified in the form. """
    if request.method == 'GET':
        # Display all the images stored under the set_name folder
        if request.args.get('set_name') != None:
            #return 'Display images from %s' % request.args.get('set_name')
            set_directory = UPLOAD_FOLDER + '/' + request.args.get('set_name')
            files = [f for f in listdir(set_directory) if isfile(join(set_directory, f))]
            imageHTML = """
                <!doctype html>
                <title>""" + request.args.get('set_name') + \
                """ Set</title> """
            for image in files:
                imageHTML = imageHTML + '<img src="' + request.args.get('set_name') + \
                    '/' + image + '">'
            return imageHTML
        else:
        # Display the file upload form
            return '''
            <!doctype html>
            <title>Upload new image</title>
            <h1>Upload new image</h1>
            <form method=post enctype=multipart/form-data>
                <p><input type=file name=file>
                <input type=submit value=Upload></p>
            </form>
            '''
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

@APP.route('/images/<image>')
def serve_image(set_name, image):
    return APP.send_static_file(UPLOAD_FOLDER + '/' + set_name + '/' + image)

# Check if model has update
@APP.route('/push-update')
def push_update():
	return APP.send_static_file('README.md')

@APP.route('/check-version')
def check_version():
	version = request.args.get('version')
	
	file_info = os.stat('static/README.md')
	size = file_info[ST_SIZE]
	mod_time = time.asctime(time.localtime(file_info[ST_MTIME]))
	return 'file size: ' + str(size) + '<br>' + 'mod time: ' + str(mod_time)
