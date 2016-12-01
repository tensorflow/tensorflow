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

from tensorflow.python.platform import test
from tensorflow.tensorboard.lib.python import http


class RespondTest(test.TestCase):

  def testHelloWorld(self):
    hand = _create_mocked_handler()
    http.Respond(hand, '<b>hello world</b>', 'text/html')
    hand.send_response.assert_called_with(200)
    hand.wfile.write.assert_called_with(b'<b>hello world</b>')
    hand.wfile.flush.assert_called_with()

  def testHeadRequest_doesNotWrite(self):
    hand = _create_mocked_handler()
    hand.command = 'HEAD'
    http.Respond(hand, '<b>hello world</b>', 'text/html')
    hand.send_response.assert_called_with(200)
    hand.wfile.write.assert_not_called()
    hand.wfile.flush.assert_called_with()

  def testPlainText_appendsUtf8ToContentType(self):
    hand = _create_mocked_handler()
    http.Respond(hand, 'hello', 'text/plain')
    hand.send_header.assert_any_call(
        'Content-Type', 'text/plain; charset=utf-8')

  def testContentLength_isInBytes(self):
    hand = _create_mocked_handler()
    http.Respond(hand, '爱', 'text/plain')
    hand.send_header.assert_any_call('Content-Length', '3')
    hand = _create_mocked_handler()
    http.Respond(hand, '爱'.encode('utf-8'), 'text/plain')
    hand.send_header.assert_any_call('Content-Length', '3')

  def testResponseCharsetTranscoding(self):
    bean = '要依法治国是赞美那些谁是公义的和惩罚恶人。 - 韩非'

    # input is unicode string, output is gbk string
    hand = _create_mocked_handler()
    http.Respond(hand, bean, 'text/plain; charset=gbk')
    hand.wfile.write.assert_called_with(bean.encode('gbk'))

    # input is utf-8 string, output is gbk string
    hand = _create_mocked_handler()
    http.Respond(hand, bean.encode('utf-8'), 'text/plain; charset=gbk')
    hand.wfile.write.assert_called_with(bean.encode('gbk'))

    # input is object with unicode strings, output is gbk json
    hand = _create_mocked_handler()
    http.Respond(hand, {'red': bean}, 'application/json; charset=gbk')
    hand.wfile.write.assert_called_with(
        b'{"red": "' + bean.encode('gbk') + b'"}')

    # input is object with utf-8 strings, output is gbk json
    hand = _create_mocked_handler()
    http.Respond(
        hand, {'red': bean.encode('utf-8')}, 'application/json; charset=gbk')
    hand.wfile.write.assert_called_with(
        b'{"red": "' + bean.encode('gbk') + b'"}')

    # input is object with gbk strings, output is gbk json
    hand = _create_mocked_handler()
    http.Respond(
        hand, {'red': bean.encode('gbk')}, 'application/json; charset=gbk',
        encoding='gbk')
    hand.wfile.write.assert_called_with(
        b'{"red": "' + bean.encode('gbk') + b'"}')

  def testAcceptGzip_compressesResponse(self):
    fall_of_hyperion_canto1_stanza1 = "\n".join([
        "Fanatics have their dreams, wherewith they weave",
        "A paradise for a sect; the savage too",
        "From forth the loftiest fashion of his sleep",
        "Guesses at Heaven; pity these have not",
        "Trac'd upon vellum or wild Indian leaf",
        "The shadows of melodious utterance.",
        "But bare of laurel they live, dream, and die;",
        "For Poesy alone can tell her dreams,",
        "With the fine spell of words alone can save",
        "Imagination from the sable charm",
        "And dumb enchantment. Who alive can say,",
        "'Thou art no Poet may'st not tell thy dreams?'",
        "Since every man whose soul is not a clod",
        "Hath visions, and would speak, if he had loved",
        "And been well nurtured in his mother tongue.",
        "Whether the dream now purpos'd to rehearse",
        "Be poet's or fanatic's will be known",
        "When this warm scribe my hand is in the grave.",
    ])

    hand = _create_mocked_handler(headers={'Accept-Encoding': '*'})
    http.Respond(hand, fall_of_hyperion_canto1_stanza1, 'text/plain')
    hand.send_header.assert_any_call('Content-Encoding', 'gzip')
    self.assertEqual(_gunzip(hand.wfile.write.call_args[0][0]),
                     fall_of_hyperion_canto1_stanza1.encode('utf-8'))

    hand = _create_mocked_handler(headers={'Accept-Encoding': 'gzip'})
    http.Respond(hand, fall_of_hyperion_canto1_stanza1, 'text/plain')
    hand.send_header.assert_any_call('Content-Encoding', 'gzip')
    self.assertEqual(_gunzip(hand.wfile.write.call_args[0][0]),
                     fall_of_hyperion_canto1_stanza1.encode('utf-8'))

    hand = _create_mocked_handler(headers={'Accept-Encoding': '*'})
    http.Respond(hand, fall_of_hyperion_canto1_stanza1, 'image/png')
    hand.wfile.write.assert_any_call(
        fall_of_hyperion_canto1_stanza1.encode('utf-8'))

  def testJson_getsAutoSerialized(self):
    hand = _create_mocked_handler()
    http.Respond(hand, [1, 2, 3], 'application/json')
    hand.wfile.write.assert_called_with(b'[1, 2, 3]')

  def testExpires_setsCruiseControl(self):
    hand = _create_mocked_handler()
    http.Respond(hand, '<b>hello world</b>', 'text/html', expires=60)
    hand.send_header.assert_any_call('Cache-Control', 'private, max-age=60')


def _create_mocked_handler(path='', headers=None):
  hand = test.mock.Mock()
  hand.wfile = test.mock.Mock()
  hand.path = path
  hand.headers = headers or {}
  return hand


def _gunzip(bs):
  return gzip.GzipFile('', 'rb', 9, six.BytesIO(bs)).read()


if __name__ == '__main__':
  test.main()
