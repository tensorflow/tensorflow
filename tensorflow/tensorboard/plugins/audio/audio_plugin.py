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
"""The TensorBoard Audio plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import urllib
from werkzeug import wrappers

from tensorflow.tensorboard.backend import http_util
from tensorflow.tensorboard.backend.event_processing import event_accumulator
from tensorflow.tensorboard.plugins import base_plugin

_PLUGIN_PREFIX_ROUTE = event_accumulator.AUDIO


class AudioPlugin(base_plugin.TBPlugin):
  """Audio Plugin for TensorBoard."""

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def get_plugin_apps(self, multiplexer, unused_logdir):
    self._multiplexer = multiplexer
    return {
        '/audio': self._serve_audio_metadata,
        '/individualAudio': self._serve_individual_audio,
        '/tags': self._serve_tags,
    }

  def is_active(self):
    """The audio plugin is active iff any run has at least one relevant tag."""
    return any(self.index_impl().values())

  def _index_impl(self):
    return {
        run_name: run_data[event_accumulator.AUDIO]
        for (run_name, run_data) in self._multiplexer.Runs().items()
        if event_accumulator.AUDIO in run_data
    }

  @wrappers.Request.application
  def _serve_audio_metadata(self, request):
    """Given a tag and list of runs, serve a list of metadata for audio.

    Note that the audio themselves are not sent; instead, we respond with URLs
    to the audio. The frontend should treat these URLs as opaque and should not
    try to parse information about them or generate them itself, as the format
    may change.

    Args:
      request: A werkzeug.wrappers.Request object.

    Returns:
      A werkzeug.Response application.
    """
    tag = request.args.get('tag')
    run = request.args.get('run')

    audio_list = self._multiplexer.Audio(run, tag)
    response = self._audio_response_for_run(audio_list, run, tag)
    return http_util.Respond(request, response, 'application/json')

  def _audio_response_for_run(self, run_audio, run, tag):
    """Builds a JSON-serializable object with information about run_audio.

    Args:
      run_audio: A list of event_accumulator.AudioValueEvent objects.
      run: The name of the run.
      tag: The name of the tag the audio entries all belong to.

    Returns:
      A list of dictionaries containing the wall time, step, URL, width, and
      height for each audio entry.
    """
    response = []
    for index, run_audio_clip in enumerate(run_audio):
      response.append({
          'wall_time': run_audio_clip.wall_time,
          'step': run_audio_clip.step,
          'content_type': run_audio_clip.content_type,
          'query': self._query_for_individual_audio(run, tag, index)
      })
    return response

  def _query_for_individual_audio(self, run, tag, index):
    """Builds a URL for accessing the specified audio.

    This should be kept in sync with _serve_audio_metadata. Note that the URL is
    *not* guaranteed to always return the same audio, since audio may be
    unloaded from the reservoir as new audio entries come in.

    Args:
      run: The name of the run.
      tag: The tag.
      index: The index of the audio entry. Negative values are OK.

    Returns:
      A string representation of a URL that will load the index-th sampled audio
      in the given run with the given tag.
    """
    query_string = urllib.parse.urlencode({
        'run': run,
        'tag': tag,
        'index': index
    })
    return query_string

  @wrappers.Request.application
  def _serve_individual_audio(self, request):
    """Serves an individual audio entry."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    index = int(request.args.get('index'))
    audio = self._multiplexer.Audio(run, tag)[index]
    return http_util.Respond(
        request, audio.encoded_audio_string, audio.content_type)

  @wrappers.Request.application
  def _serve_tags(self, request):
    index = self._index_impl()
    return http_util.Respond(request, index, 'application/json')
