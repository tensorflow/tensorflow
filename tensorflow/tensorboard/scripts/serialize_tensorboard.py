# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Consume and serialize all of the data from a running TensorBoard instance.

This program connects to a live TensorBoard backend at given port, and saves
all of the data to local disk JSON in a predictable format.

This makes it easy to mock out the TensorBoard backend so that the frontend
may be tested in isolation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path
import shutil
import threading
import urllib

import six
from six.moves import http_client
import tensorflow as tf

from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server

tf.flags.DEFINE_string('logdir', None, """the logdir to pass to the TensorBoard
backend; data will be read from this logdir for serialization.""")

tf.flags.DEFINE_string('target', None, """The directoy where serialized data
will be written""")

tf.flags.DEFINE_boolean('overwrite', False, """Whether to remove and overwrite
TARGET if it already exists.""")

tf.flags.DEFINE_boolean(
    'purge_orphaned_data', True, 'Whether to purge data that '
    'may have been orphaned due to TensorBoard restarts. '
    'Disabling purge_orphaned_data can be used to debug data '
    'disappearance.')
FLAGS = tf.flags.FLAGS

BAD_CHARACTERS = "#%&{}\\/<>*? $!'\":@+`|="
DEFAULT_SUFFIX = '.json'
IMAGE_SUFFIX = '.png'
AUDIO_SUFFIX = '.wav'
GRAPH_SUFFIX = '.pbtxt'


def Url(route, params):
  """Takes route and query params, and produce encoded url for that asset."""
  out = route
  if params:
    # sorting ensures a unique filename for each query
    sorted_params = sorted(six.iteritems(params))
    out += '?' + urllib.urlencode(sorted_params)
  return out


def Clean(s):
  """Clean a string so it can be used as a filepath."""
  for c in BAD_CHARACTERS:
    s = s.replace(c, '_')
  return s


class TensorBoardStaticSerializer(object):
  """Serialize all the routes from a TensorBoard server to static json."""

  def __init__(self, connection, target_path):
    self.connection = connection
    EnsureDirectoryExists(target_path)
    self.path = target_path

  def GetAndSave(self, url, save_suffix):
    """GET the given url. Serialize the result at clean path version of url."""
    self.connection.request('GET',
                            '/data/' + url,
                            headers={'content-type': 'text/plain'})
    response = self.connection.getresponse()
    file_name = Clean(url) + save_suffix
    destination = os.path.join(self.path, file_name)

    if response.status != 200:
      raise IOError(url)

    content = response.read()

    with open(destination, 'w') as f:
      f.write(content)
    return content

  def GetRouteAndSave(self, route, params=None):
    """GET given route and params. Serialize the result. Return as JSON."""
    url = Url(route, params)
    return json.loads(self.GetAndSave(url, DEFAULT_SUFFIX))

  def Run(self):
    """Serialize everything from a TensorBoard backend."""
    # get the runs object, which is an index for every tag.
    runs = self.GetRouteAndSave('runs')

    # collect sampled data.
    self.GetRouteAndSave('scalars')

    # now let's just download everything!
    for run, tag_type_to_tags in six.iteritems(runs):
      for tag_type, tags in six.iteritems(tag_type_to_tags):
        try:
          if tag_type == 'graph':
            # in this case, tags is a bool which specifies if graph is present.
            if tags:
              url = Url('graph', {'run': run})
              self.GetAndSave(url, GRAPH_SUFFIX)
          elif tag_type == 'images':
            for t in tags:
              images = self.GetRouteAndSave('images', {'run': run, 'tag': t})
              for im in images:
                url = 'individualImage?' + im['query']
                # pull down the images themselves.
                self.GetAndSave(url, IMAGE_SUFFIX)
          elif tag_type == 'audio':
            for t in tags:
              audio = self.GetRouteAndSave('audio', {'run': run, 'tag': t})
              for snd in audio:
                url = 'individualAudio?' + snd['query']
                # pull down the audio clips themselves
                self.GetAndSave(url, AUDIO_SUFFIX)
          elif tag_type == 'run_metadata':
            for t in tags:
              url = Url('run_metadata', {'run': run, 'tag': t})
              self.GetAndSave(url, GRAPH_SUFFIX)
          elif tag_type == 'firstEventTimestamp':
            pass
          else:
            for t in tags:
            # Save this, whatever it is :)
              self.GetRouteAndSave(tag_type, {'run': run, 'tag': t})
        except IOError as e:
          x = Exception('Retrieval failed for %s/%s/%s' % (tag_type, run, tags))
          six.raise_from(x, e)


def EnsureDirectoryExists(path):
  if not os.path.exists(path):
    os.makedirs(path)


def PrintAndLog(msg, lvl=tf.logging.INFO):
  tf.logging.log(lvl, msg)
  print(msg)


def main(unused_argv=None):
  target = FLAGS.target
  logdir = FLAGS.logdir
  if not target or not logdir:
    PrintAndLog('Both --target and --logdir are required.', tf.logging.ERROR)
    return -1
  if os.path.exists(target):
    if FLAGS.overwrite:
      if os.path.isdir(target):
        shutil.rmtree(target)
      else:
        os.remove(target)
    else:
      PrintAndLog('Refusing to overwrite target %s without --overwrite' %
                  target, tf.logging.ERROR)
      return -2
  path_to_run = server.ParseEventFilesSpec(FLAGS.logdir)

  PrintAndLog('About to load Multiplexer. This may take some time.')
  multiplexer = event_multiplexer.EventMultiplexer(
      size_guidance=server.TENSORBOARD_SIZE_GUIDANCE,
      purge_orphaned_data=FLAGS.purge_orphaned_data)
  server.ReloadMultiplexer(multiplexer, path_to_run)

  PrintAndLog('Multiplexer load finished. Starting TensorBoard server.')
  s = server.BuildServer(multiplexer, 'localhost', 0)
  server_thread = threading.Thread(target=s.serve_forever)
  server_thread.daemon = True
  server_thread.start()
  connection = http_client.HTTPConnection('localhost', s.server_address[1])

  PrintAndLog('Server setup! Downloading data from the server.')
  x = TensorBoardStaticSerializer(connection, target)
  x.Run()

  PrintAndLog('Done downloading data.')
  connection.close()
  s.shutdown()
  s.server_close()


if __name__ == '__main__':
  tf.app.run()
