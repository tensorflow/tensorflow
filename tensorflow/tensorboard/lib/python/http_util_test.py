# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests HTTP utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip

import six
from werkzeug import test as wtest
from werkzeug import wrappers
from tensorflow.python.platform import test
from tensorflow.tensorboard.lib.python import http_util


class RespondTest(test.TestCase):

  def testHelloWorld(self):
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, '<b>hello world</b>', 'text/html')
    self.assertEqual(r.status_code, 200)
    self.assertEqual(r.response[0], six.b('<b>hello world</b>'))

  def testHeadRequest_doesNotWrite(self):
    builder = wtest.EnvironBuilder(method='HEAD')
    env = builder.get_environ()
    request = wrappers.Request(env)
    r = http_util.Respond(request, '<b>hello world</b>', 'text/html')
    self.assertEqual(r.status_code, 200)
    self.assertEqual(r.response[0], six.b(''))

  def testPlainText_appendsUtf8ToContentType(self):
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, 'hello', 'text/plain')
    h = r.headers
    self.assertEqual(h.get('Content-Type'), 'text/plain; charset=utf-8')

  def testContentLength_isInBytes(self):
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, '爱', 'text/plain')
    self.assertEqual(r.headers.get('Content-Length'), '3')
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, '爱'.encode('utf-8'), 'text/plain')
    self.assertEqual(r.headers.get('Content-Length'), '3')

  def testResponseCharsetTranscoding(self):
    bean = '要依法治国是赞美那些谁是公义的和惩罚恶人。 - 韩非'

    # input is unicode string, output is gbk string
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, bean, 'text/plain; charset=gbk')
    self.assertEqual(r.response[0], bean.encode('gbk'))

    # input is utf-8 string, output is gbk string
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, bean.encode('utf-8'), 'text/plain; charset=gbk')
    self.assertEqual(r.response[0], bean.encode('gbk'))

    # input is object with unicode strings, output is gbk json
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, {'red': bean}, 'application/json; charset=gbk')
    self.assertEqual(r.response[0], b'{"red": "' + bean.encode('gbk') + b'"}')

    # input is object with utf-8 strings, output is gbk json
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(
        q, {'red': bean.encode('utf-8')}, 'application/json; charset=gbk')
    self.assertEqual(r.response[0], b'{"red": "' + bean.encode('gbk') + b'"}')

    # input is object with gbk strings, output is gbk json
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(
        q, {'red': bean.encode('gbk')},
        'application/json; charset=gbk',
        encoding='gbk')
    self.assertEqual(r.response[0], b'{"red": "' + bean.encode('gbk') + b'"}')

  def testAcceptGzip_compressesResponse(self):
    fall_of_hyperion_canto1_stanza1 = '\n'.join([
        'Fanatics have their dreams, wherewith they weave',
        'A paradise for a sect; the savage too',
        'From forth the loftiest fashion of his sleep',
        'Guesses at Heaven; pity these have not',
        'Trac\'d upon vellum or wild Indian leaf',
        'The shadows of melodious utterance.',
        'But bare of laurel they live, dream, and die;',
        'For Poesy alone can tell her dreams,',
        'With the fine spell of words alone can save',
        'Imagination from the sable charm',
        'And dumb enchantment. Who alive can say,',
        '\'Thou art no Poet may\'st not tell thy dreams?\'',
        'Since every man whose soul is not a clod',
        'Hath visions, and would speak, if he had loved',
        'And been well nurtured in his mother tongue.',
        'Whether the dream now purpos\'d to rehearse',
        'Be poet\'s or fanatic\'s will be known',
        'When this warm scribe my hand is in the grave.',
    ])

    e1 = wtest.EnvironBuilder(headers={'Accept-Encoding': '*'}).get_environ()
    any_encoding = wrappers.Request(e1)

    r = http_util.Respond(
        any_encoding, fall_of_hyperion_canto1_stanza1, 'text/plain')
    self.assertEqual(r.headers.get('Content-Encoding'), 'gzip')

    self.assertEqual(
        _gunzip(r.response[0]), fall_of_hyperion_canto1_stanza1.encode('utf-8'))

    e2 = wtest.EnvironBuilder(headers={'Accept-Encoding': 'gzip'}).get_environ()
    gzip_encoding = wrappers.Request(e2)

    r = http_util.Respond(
        gzip_encoding, fall_of_hyperion_canto1_stanza1, 'text/plain')
    self.assertEqual(r.headers.get('Content-Encoding'), 'gzip')
    self.assertEqual(
        _gunzip(r.response[0]), fall_of_hyperion_canto1_stanza1.encode('utf-8'))

    r = http_util.Respond(
        any_encoding, fall_of_hyperion_canto1_stanza1, 'image/png')
    self.assertEqual(
        r.response[0], fall_of_hyperion_canto1_stanza1.encode('utf-8'))

  def testJson_getsAutoSerialized(self):
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, [1, 2, 3], 'application/json')
    self.assertEqual(r.response[0], b'[1, 2, 3]')

  def testExpires_setsCruiseControl(self):
    q = wrappers.Request(wtest.EnvironBuilder().get_environ())
    r = http_util.Respond(q, '<b>hello world</b>', 'text/html', expires=60)
    self.assertEqual(r.headers.get('Cache-Control'), 'private, max-age=60')


def _gunzip(bs):
  return gzip.GzipFile('', 'rb', 9, six.BytesIO(bs)).read()


if __name__ == '__main__':
  test.main()
