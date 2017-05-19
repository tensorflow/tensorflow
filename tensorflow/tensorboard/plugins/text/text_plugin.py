# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""The TensorBoard Text plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import textwrap

# pylint: disable=g-bad-import-order
# Necessary for an internal test with special behavior for numpy.
import numpy as np
# pylint: enable=g-bad-import-order

import bleach
# pylint: disable=g-bad-import-order
# Google-only: import markdown_freewisdom
import markdown
import six
# pylint: enable=g-bad-import-order
from werkzeug import wrappers

from tensorflow.python.framework import tensor_util
from tensorflow.python.summary import text_summary
from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.plugins import base_plugin

# The prefix of routes provided by this plugin.
_PLUGIN_PREFIX_ROUTE = 'text'

# HTTP routes
RUNS_ROUTE = '/runs'
TEXT_ROUTE = '/text'

ALLOWED_TAGS = [
    'ul',
    'ol',
    'li',
    'p',
    'pre',
    'code',
    'blockquote',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'hr',
    'br',
    'strong',
    'em',
    'a',
    'img',
    'table',
    'thead',
    'tbody',
    'td',
    'tr',
    'th',
]

ALLOWED_ATTRIBUTES = {'a': ['href', 'title'], 'img': ['src', 'title', 'alt']}

WARNING_TEMPLATE = textwrap.dedent("""\
  **Warning:** This text summary contained data of dimensionality %d, but only \
  2d tables are supported. Showing a 2d slice of the data instead.""")


def markdown_and_sanitize(markdown_string):
  """Takes a markdown string and converts it into sanitized html.

  It uses the table extension; while that's not a part of standard
  markdown, it is sure to be useful for TensorBoard users.

  The sanitizer uses the allowed_tags and attributes specified above. Mostly,
  we ensure that our standard use cases like tables and links are supported.

  Args:
    markdown_string: Markdown string to sanitize

  Returns:
    a string containing sanitized html for input markdown
  """
  # Convert to utf-8 whenever we have a binary input.
  if isinstance(markdown_string, six.binary_type):
    markdown_string = markdown_string.decode('utf-8')

  string_html = markdown.markdown(
      markdown_string, extensions=['markdown.extensions.tables'])
  string_sanitized = bleach.clean(
      string_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES)
  return string_sanitized


def make_table_row(contents, tag='td'):
  """Given an iterable of string contents, make a table row.

  Args:
    contents: An iterable yielding strings.
    tag: The tag to place contents in. Defaults to 'td', you might want 'th'.

  Returns:
    A string containing the content strings, organized into a table row.

  Example: make_table_row(['one', 'two', 'three']) == '''
  <tr>
  <td>one</td>
  <td>two</td>
  <td>three</td>
  </tr>'''
  """
  columns = ('<%s>%s</%s>\n' % (tag, s, tag) for s in contents)
  return '<tr>\n' + ''.join(columns) + '</tr>\n'


def make_table(contents, headers=None):
  """Given a numpy ndarray of strings, concatenate them into a html table.

  Args:
    contents: A np.ndarray of strings. May be 1d or 2d. In the 1d case, the
      table is laid out vertically (i.e. row-major).
    headers: A np.ndarray or list of string header names for the table.

  Returns:
    A string containing all of the content strings, organized into a table.

  Raises:
    ValueError: If contents is not a np.ndarray.
    ValueError: If contents is not 1d or 2d.
    ValueError: If contents is empty.
    ValueError: If headers is present and not a list, tuple, or ndarray.
    ValueError: If headers is not 1d.
    ValueError: If number of elements in headers does not correspond to number
      of columns in contents.
  """
  if not isinstance(contents, np.ndarray):
    raise ValueError('make_table contents must be a numpy ndarray')

  if contents.ndim not in [1, 2]:
    raise ValueError('make_table requires a 1d or 2d numpy array, was %dd' %
                     contents.ndim)

  if headers:
    if isinstance(headers, list) or isinstance(headers, tuple):
      headers = np.array(headers)
    if not isinstance(headers, np.ndarray):
      raise ValueError('Could not convert headers %s into np.ndarray' % headers)
    if headers.ndim != 1:
      raise ValueError('Headers must be 1d, is %dd' % headers.ndim)
    expected_n_columns = contents.shape[1] if contents.ndim == 2 else 1
    if headers.shape[0] != expected_n_columns:
      raise ValueError('Number of headers %d must match number of columns %d' %
                       (headers.shape[0], expected_n_columns))
    header = '<thead>\n%s</thead>\n' % make_table_row(headers, tag='th')
  else:
    header = ''

  n_rows = contents.shape[0]
  if contents.ndim == 1:
    # If it's a vector, we need to wrap each element in a new list, otherwise
    # we would turn the string itself into a row (see test code)
    rows = (make_table_row([contents[i]]) for i in range(n_rows))
  else:
    rows = (make_table_row(contents[i, :]) for i in range(n_rows))

  return '<table>\n%s<tbody>\n%s</tbody>\n</table>' % (header, ''.join(rows))


def reduce_to_2d(arr):
  """Given a np.npdarray with nDims > 2, reduce it to 2d.

  It does this by selecting the zeroth coordinate for every dimension greater
  than two.

  Args:
    arr: a numpy ndarray of dimension at least 2.

  Returns:
    A two-dimensional subarray from the input array.

  Raises:
    ValueError: If the argument is not a numpy ndarray, or the dimensionality
      is too low.
  """
  if not isinstance(arr, np.ndarray):
    raise ValueError('reduce_to_2d requires a numpy.ndarray')

  ndims = len(arr.shape)
  if ndims < 2:
    raise ValueError('reduce_to_2d requires an array of dimensionality >=2')
  # slice(None) is equivalent to `:`, so we take arr[0,0,...0,:,:]
  slices = ([0] * (ndims - 2)) + [slice(None), slice(None)]
  return arr[slices]


def text_array_to_html(text_arr):
  """Take a numpy.ndarray containing strings, and convert it into html.

  If the ndarray contains a single scalar string, that string is converted to
  html via our sanitized markdown parser. If it contains an array of strings,
  the strings are individually converted to html and then composed into a table
  using make_table. If the array contains dimensionality greater than 2,
  all but two of the dimensions are removed, and a warning message is prefixed
  to the table.

  Args:
    text_arr: A numpy.ndarray containing strings.

  Returns:
    The array converted to html.
  """
  if not text_arr.shape:
    # It is a scalar. No need to put it in a table, just apply markdown
    return markdown_and_sanitize(text_arr.astype(np.dtype(str)).tostring())
  warning = ''
  if len(text_arr.shape) > 2:
    warning = markdown_and_sanitize(WARNING_TEMPLATE % len(text_arr.shape))
    text_arr = reduce_to_2d(text_arr)

  html_arr = [markdown_and_sanitize(x) for x in text_arr.reshape(-1)]
  html_arr = np.array(html_arr).reshape(text_arr.shape)

  return warning + make_table(html_arr)


def process_string_tensor_event(event):
  """Convert a TensorEvent into a JSON-compatible response."""
  string_arr = tensor_util.MakeNdarray(event.tensor_proto)
  html = text_array_to_html(string_arr)
  return {
      'wall_time': event.wall_time,
      'step': event.step,
      'text': html,
  }


class TextPlugin(base_plugin.TBPlugin):
  """Text Plugin for TensorBoard."""

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def index_impl(self):
    run_to_series = {}
    name = text_summary.TextSummaryPluginAsset.plugin_name
    run_to_assets = self.multiplexer.PluginAssets(name)

    for run, assets in run_to_assets.items():
      if 'tensors.json' in assets:
        tensors_json = self.multiplexer.RetrievePluginAsset(
            run, name, 'tensors.json')
        tensors = json.loads(tensors_json)
        run_to_series[run] = tensors
    return run_to_series

  @wrappers.Request.application
  def runs_route(self, request):
    index = self.index_impl()
    return http_util.Respond(request, index, 'application/json')

  def text_impl(self, run, tag):
    try:
      text_events = self.multiplexer.Tensors(run, tag)
    except KeyError:
      text_events = []
    responses = [process_string_tensor_event(ev) for ev in text_events]
    return responses

  @wrappers.Request.application
  def text_route(self, request):
    run = request.args.get('run')
    tag = request.args.get('tag')
    response = self.text_impl(run, tag)
    return http_util.Respond(request, response, 'application/json')

  def get_plugin_apps(self, multiplexer, unused_logdir):
    self.multiplexer = multiplexer
    return {
        RUNS_ROUTE: self.runs_route,
        TEXT_ROUTE: self.text_route,
    }

  def is_active(self):
    """Determines whether this plugin is active.

    This plugin is only active if TensorBoard sampled any text summaries.

    Returns:
      Whether this plugin is active.
    """
    return bool(self.index_impl())
