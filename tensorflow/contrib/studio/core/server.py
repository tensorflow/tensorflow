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
"""Module for building TensorFlow Studio servers.

This is its own module so it can be used in both actual code and test code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import threading
import time
import urlparse

import six
from six.moves import BaseHTTPServer
from six.moves import socketserver

from tensorflow.python.platform import logging


class ThreadedHTTPServer(socketserver.ThreadingMixIn,
                         BaseHTTPServer.HTTPServer):
  """A threaded HTTP server."""
  daemon = True


class StudioHandler(BaseHTTPServer.BaseHTTPRequestHandler):

  def do_GET(self):  # pylint: disable=invalid-name
    """Handler for all get requests."""
    parsed_url = urlparse.urlparse(self.path)

    # Remove a trailing slash, if present.
    clean_path = parsed_url.path
    if clean_path.endswith('/'):
      clean_path = clean_path[:-1]

    query_params = urlparse.parse_qs(parsed_url.query)
    # parse_qs returns a list of values for each key; we're only interested in
    # the first.
    for key in query_params:
      value_count = len(query_params[key])
      if value_count != 1:
        self.send_error(
            400, 'query parameter %s should have exactly one value, had %d' %
            (key, value_count))
        return
      query_params[key] = query_params[key][0]

    self.send_error(404, 'No handlers')


def BuildServer(working_dir, host, port):
  """Sets up an HTTP server for running TensorBoard.

  Args:
    working_dir: Working directory.
    host: The host name.
    port: The port number to bind to, or 0 to pick one automatically.

  Returns:
    A `BaseHTTPServer.HTTPServer`.
  """
  return ThreadedHTTPServer((host, port), StudioHandler)

