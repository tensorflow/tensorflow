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
"""Module for building TensorBoard servers.

This is its own module so it can be used in both actual code and test code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import threading
import time

import six
from six.moves import BaseHTTPServer
from six.moves import socketserver

from tensorflow.python.platform import logging
from tensorflow.python.summary import event_accumulator
from tensorflow.tensorboard.backend import tensorboard_handler

# How many elements to store per tag, by tag type
TENSORBOARD_SIZE_GUIDANCE = {
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.SCALARS: 1000,
    event_accumulator.HISTOGRAMS: 1,
}

# How often to reload new data after the latest load (secs)
LOAD_INTERVAL = 60


def ParseEventFilesSpec(logdir):
  """Parses `logdir` into a map from paths to run group names.

  The events files flag format is a comma-separated list of path specifications.
  A path specification either looks like 'group_name:/path/to/directory' or
  '/path/to/directory'; in the latter case, the group is unnamed. Group names
  cannot start with a forward slash: /foo:bar/baz will be interpreted as a
  spec with no name and path '/foo:bar/baz'.

  Globs are not supported.

  Args:
    logdir: A comma-separated list of run specifications.
  Returns:
    A dict mapping directory paths to names like {'/path/to/directory': 'name'}.
    Groups without an explicit name are named after their path. If logdir is
    None, returns an empty dict, which is helpful for testing things that don't
    require any valid runs.
  """
  files = {}
  if logdir is None:
    return files
  for specification in logdir.split(','):
    # If the spec looks like /foo:bar/baz, then we assume it's a path with a
    # colon.
    if ':' in specification and specification[0] != '/':
      # We split at most once so run_name:/path:with/a/colon will work.
      run_name, path = specification.split(':', 1)
    else:
      run_name = None
      path = specification

    if not os.path.isabs(path):
      # Create absolute path out of relative one.
      path = os.path.join(os.path.realpath('.'), path)

    files[path] = run_name
  return files


def ReloadMultiplexer(multiplexer, path_to_run):
  """Loads all runs into the multiplexer.

  Args:
    multiplexer: The `EventMultiplexer` to add runs to and reload.
    path_to_run: A dict mapping from paths to run names, where `None` as the run
      name is interpreted as a run name equal to the path.
  """
  start = time.time()
  for (path, name) in six.iteritems(path_to_run):
    multiplexer.AddRunsFromDirectory(path, name)
  multiplexer.Reload()
  duration = time.time() - start
  logging.info('Multiplexer done loading. Load took %0.1f secs', duration)


def StartMultiplexerReloadingThread(multiplexer, path_to_run,
                                    load_interval=LOAD_INTERVAL):
  """Starts a thread to automatically reload the given multiplexer.

  The thread will reload the multiplexer by calling `ReloadMultiplexer` every
  `load_interval` seconds, starting immediately.

  Args:
    multiplexer: The `EventMultiplexer` to add runs to and reload.
    path_to_run: A dict mapping from paths to run names, where `None` as the run
      name is interpreted as a run name equal to the path.
    load_interval: How many seconds to wait after one load before starting the
      next load.

  Returns:
    A started `threading.Thread` that reloads the multiplexer.

  """
  # Ensure the Multiplexer initializes in a loaded state before it adds runs
  # So it can handle HTTP requests while runs are loading
  multiplexer.Reload()

  def _ReloadForever():
    while True:
      ReloadMultiplexer(multiplexer, path_to_run)
      time.sleep(load_interval)

  thread = threading.Thread(target=_ReloadForever)
  thread.daemon = True
  thread.start()
  return thread


class ThreadedHTTPServer(socketserver.ThreadingMixIn,
                         BaseHTTPServer.HTTPServer):

  """A threaded HTTP server."""
  daemon = True


def BuildServer(multiplexer, host, port):
  """Sets up an HTTP server for running TensorBoard.

  Args:
    multiplexer: An `EventMultiplexer` that the server will query for
      information about events.
    host: The host name.
    port: The port number to bind to, or 0 to pick one automatically.

  Returns:
    A `BaseHTTPServer.HTTPServer`.
  """
  factory = functools.partial(tensorboard_handler.TensorboardHandler,
                              multiplexer)
  return ThreadedHTTPServer((host, port), factory)
