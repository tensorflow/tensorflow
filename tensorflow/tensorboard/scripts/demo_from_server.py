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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import urllib2
import requests
import os.path
import shutil
import six


class TensorBoardStaticSerializer(object):
  """Serialize all the routes from a TensorBoard server to static json."""

  def __init__(self, host, port, path):
    self.server_address = '%s:%d/' % (host, port)
    EnsureDirectoryExists(path)
    self.path = path
    self.img_id = 0

  def _QuoteOrNone(self, x):
    if x is None:
      return None
    else:
      return urllib2.quote(x)

  def _RetrieveRoute(self, route, run=None, tag=None):
    """Load route (possibly with run and tag), return the json."""
    r = self._SendRequest(route, run, tag)
    j = r.json()
    return j

  def _SendRequest(self, route, run=None, tag=None):
    url = self.server_address + route
    run = self._QuoteOrNone(run)
    tag = self._QuoteOrNone(tag)
    if run is not None:
      url += '?run={}'.format(run)
      if tag is not None:
        url += '&tag={}'.format(tag)
    r = requests.get(url)
    if r.status_code != 200:
      raise IOError
    return r

  def _SaveRouteJsonToDisk(self, data, route, run=None, tag=None):
    """Save the route, run, tag result to a predictable spot on disk."""
    print('%s/%s/%s' % (route, run, tag))
    if run is not None:
      run = run.replace(' ', '_')
    if tag is not None:
      tag = tag.replace(' ', '_')
      tag = tag.replace('(', '_')
      tag = tag.replace(')', '_')
    components = [x for x in [self.path, route, run, tag] if x]
    path = os.path.join(*components) + '.json'
    EnsureDirectoryExists(os.path.dirname(path))
    with open(path, 'w') as f:
      json.dump(data, f)

  def _RetrieveAndSave(self, route, run=None, tag=None):
    """Retrieve data, and save it to disk."""
    data = self._RetrieveRoute(route, run, tag)
    self._SaveRouteJsonToDisk(data, route, run, tag)
    return data

  def _SerializeImages(self, run, tag):
    """Serialize all the images, and use ids not query parameters."""
    EnsureDirectoryExists(os.path.join(self.path, 'individualImage'))
    images = self._RetrieveRoute('images', run, tag)
    for im in images:
      q = im['query']
      im['query'] = self.img_id
      path = '%s/individualImage/%d.png' % (self.path, self.img_id)
      self.img_id += 1
      r = requests.get(self.server_address + 'individualImage?' + q)
      if r.status_code != 200:
        raise IOError
      with open(path, 'wb') as f:
        f.write(r.content)
    self._SaveRouteJsonToDisk(images, 'images', run, tag)


  def Run(self):
    """Main method that loads and serializes everything."""
    runs = self._RetrieveAndSave('runs')
    for run, tag_type_to_tags in six.iteritems(runs):
      for tag_type, tags in six.iteritems(tag_type_to_tags):
        try:
          if tag_type == 'graph':
            if tags:
              r = self._SendRequest('graph', run, None)
              pbtxt = r.text
              fname = run.replace(' ', '_') + '.pbtxt'
              path = os.path.join(self.path, 'graph', fname)
              EnsureDirectoryExists(os.path.dirname(path))
              with open(path, 'w') as f:
                f.write(pbtxt)
          elif tag_type == 'images':
            for t in tags:
              self._SerializeImages(run, t)
          else:
            for t in tags:
              self._RetrieveAndSave(tag_type, run, t)
        except requests.exceptions.ConnectionError as e:
          print('Retrieval failed for %s/%s/%s' % (tag_type, run, tag))
          print('Got error: ', e)
          print('continuing...')
          continue
        except IOError as e:
          print('Retrieval failed for %s/%s/%s' % (tag_type, run, tag))
          print('Got error: ', e)
          print('continuing...')
          continue


def EnsureDirectoryExists(path):
  if not os.path.exists(path):
    os.makedirs(path)

def main(unused_argv=None):
  target = '/tmp/tensorboard_demo_data'
  port = 6006
  host = 'http://localhost'
  if os.path.exists(target):
    if os.path.isdir(target):
      shutil.rmtree(target)
    else:
      os.remove(target)
  x = TensorBoardStaticSerializer(host, port, target)
  x.Run()

if __name__ == '__main__':
  main()
