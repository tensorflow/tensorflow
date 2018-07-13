# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Function for interpolating formatted errors from the TensorFlow runtime.

Exposes the function `interpolate` to interpolate messages with tags of the form
^^type:name:format^^.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import re
import string

import six

_NAME_REGEX = r"[A-Za-z0-9.][A-Za-z0-9_.\-/]*?"
_FORMAT_REGEX = r"[A-Za-z0-9_.\-/${}:]+"
_TAG_REGEX = r"\^\^({name}):({name}):({fmt})\^\^".format(
    name=_NAME_REGEX, fmt=_FORMAT_REGEX)
_INTERPOLATION_REGEX = r"^(.*?)({tag})".format(tag=_TAG_REGEX)
_INTERPOLATION_PATTERN = re.compile(_INTERPOLATION_REGEX)

_ParseTag = collections.namedtuple("_ParseTag", ["type", "name", "format"])


def _parse_message(message):
  """Parses the message.

  Splits the message into separators and tags. Tags are named tuples
  representing the string ^^type:name:format^^ and they are separated by
  separators. For example, in
  "123^^node:Foo:${file}^^456^^node:Bar:${line}^^789", there are two tags and
  three separators. The separators are the numeric characters.

  Args:
    message: String to parse

  Returns:
    (list of separator strings, list of _ParseTags).

    For example, if message is "123^^node:Foo:${file}^^456" then this function
    returns (["123", "456"], [_ParseTag("node", "Foo", "${file}")])
  """
  seps = []
  tags = []
  pos = 0
  while pos < len(message):
    match = re.match(_INTERPOLATION_PATTERN, message[pos:])
    if match:
      seps.append(match.group(1))
      tags.append(_ParseTag(match.group(3), match.group(4), match.group(5)))
      pos += match.end()
    else:
      break
  seps.append(message[pos:])
  return seps, tags


# TODO(jtkeeling): Modify to actually interpolate format strings rather than
# echoing them.
def interpolate(error_message):
  """Interpolates an error message.

  The error message can contain tags of the form ^^type:name:format^^ which will
  be replaced.

  Args:
    error_message: A string to interpolate.

  Returns:
    The string with tags of the form ^^type:name:format^^ interpolated.
  """
  seps, tags = _parse_message(error_message)
  subs = [string.Template(tag.format).safe_substitute({}) for tag in tags]
  return "".join(
      itertools.chain(*six.moves.zip_longest(seps, subs, fillvalue="")))
