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
# pylint: disable=g-short-docstring-punctuation
"""Working with audio using FFmpeg.

See the @{$python/contrib.ffmpeg} guide.

@@decode_audio
@@encode_audio
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.ffmpeg.ffmpeg_ops import decode_audio
from tensorflow.contrib.ffmpeg.ffmpeg_ops import decode_video
from tensorflow.contrib.ffmpeg.ffmpeg_ops import encode_audio

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['decode_audio', 'encode_audio', 'decode_video']
remove_undocumented(__name__, _allowed_symbols)
