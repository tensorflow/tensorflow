# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Http utilities for various handlers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
import imghdr
import json
import mimetypes
import os

from six import BytesIO
from six.moves import BaseHTTPServer
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves.urllib import parse as urlparse

from tensorflow.python.platform import logging
from tensorflow.python.platform import resource_loader


def path_is_safe(path):
    """Check path is safe (stays within current directory).

    This is for preventing directory-traversal attacks.

    Args:
      path: The path to check for safety.

    Returns:
      True if the given path stays within the current directory, and false
      if it would escape to a higher directory. E.g. _path_is_safe('index.html')
      returns true, but _path_is_safe('../../../etc/password') returns false.
    """
    base = os.path.abspath(os.curdir)
    absolute_path = os.path.abspath(path)
    prefix = os.path.commonprefix([base, absolute_path])
    return prefix == base


def serve_static_file(handler, path):
    """Serves the static file located at the given path.

    Args:
      handler: `HttpHandler` object to write reponse into.
      path: The path of the static file, relative to the contrib/studio/ directory.
    """
    # Strip off the leading forward slash.
    path = path.lstrip('/')
    if not path_is_safe(path):
      logging.info('path %s not safe, sending 404', path)
      # Traversal attack, so 404.
      handler.send_error(404)
      return

    if path.startswith('external'):
      # For compatibility with latest version of Bazel, we renamed bower
      # packages to use '_' rather than '-' in their package name.
      # This means that the directory structure is changed too.
      # So that all our recursive imports work, we need to modify incoming
      # requests to map onto the new directory structure.
      components = path.split('/')
      components[1] = components[1].replace('-', '_')
      path = ('/').join(components)
      path = os.path.join('../', path)
    else:
      path = os.path.join('contrib/studio', path)
    # Open the file and read it.
    try:
      contents = resource_loader.load_resource(path)
    except IOError:
      logging.info('path %s not found, sending 404', path)
      handler.send_error(404)
      return

    handler.send_response(200)

    mimetype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    handler.send_header('Content-Type', mimetype)
    handler.end_headers()
    handler.wfile.write(contents)

