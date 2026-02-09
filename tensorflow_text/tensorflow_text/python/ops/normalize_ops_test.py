# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

# coding=utf-8
"""Tests for normalization ops in tensorflow_text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import normalize_ops


def _Utf8(char):
  return char.encode("utf-8")


@test_util.run_all_in_graph_and_eager_modes
class NormalizeOpsTest(test.TestCase):

  def test_lowercase_one_string(self):
    txt = [
        " TExt to loWERcase! ",
    ]
    expected = [
        b" text to lowercase! ",
    ]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_lowercase_text(self):
    txt = [
        "Punctuation and digits: -*/+$#%@%$123456789#^$*%&",
        "Non-latin UTF8 chars: ΘͽʦȺЩ",
        "Accented chars: ĎÔPQRŔSŠoóôpqrŕsštťuúvwxyý",
        "Non-UTF8-letters: e.g. ◆, ♥, and the emoji symbol ( ͡° ͜ʖ ͡°)",
        "Folded: ßς", ""
    ]
    expected = [
        _Utf8(u"punctuation and digits: -*/+$#%@%$123456789#^$*%&"),
        _Utf8(u"non-latin utf8 chars: θͽʦⱥщ"),
        _Utf8(u"accented chars: ďôpqrŕsšoóôpqrŕsštťuúvwxyý"),
        _Utf8(
            u"non-utf8-letters: e.g. ◆, ♥, and the emoji symbol ( ͡° ͜ʖ ͡°)"
        ),
        _Utf8(u"folded: ssσ"), b""
    ]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_lowercase_one_string_ragged(self):
    txt = ragged_factory_ops.constant([[" TExt ", "to", " loWERcase! "],
                                       [" TExt to loWERcase! "]])
    expected = [[b" text ", b"to", b" lowercase! "], [b" text to lowercase! "]]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_lowercase_empty_string(self):
    txt = [
        "",
    ]
    expected = [
        b"",
    ]
    self.assertAllEqual(expected, normalize_ops.case_fold_utf8(txt))

  def test_normalize_nfkc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"ṩ".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFKC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfkc"))

  def test_normalize_nfkc_batch(self):
    txt = [
        u"\u1e9b\u0323",
        u"\ufb01",
    ]
    expected = [
        b"\xe1\xb9\xa9",
        b"fi",
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, u"NFKC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, u"nfkc"))

  def test_normalize_nfkc_ragged(self):
    txt = ragged_factory_ops.constant([[[u"\u1e9b\u0323 \ufb01"], []],
                                       [[u"\u1e9b\u0323", u"\ufb01"]]])
    expected = [[[u"ṩ fi".encode("utf-8")], []],
                [[u"ṩ".encode("utf-8"), b"fi"]]]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFKC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfkc"))

  def test_normalize_nfc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"\u1e9b\u0323".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFC"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfc"))

  def test_normalize_nfd(self):
    txt = [u"\u1e9b\u0323"]
    expected = [
        u"\u017f\u0323\u0307".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFD"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfd"))

  def test_normalize_nfkd(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"\u0073\u0323\u0307".encode("utf-8"),
    ]
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "NFKD"))
    self.assertAllEqual(expected, normalize_ops.normalize_utf8(txt, "nfkd"))

  def test_unknown_normalization_form(self):
    with self.assertRaises(errors.InvalidArgumentError):
      bomb = normalize_ops.normalize_utf8(["cant readme", "wont read me"],
                                          "cantfindme")
      self.evaluate(bomb)


@test_util.run_all_in_graph_and_eager_modes
class NormalizeWithOffsetsMapOpsTest(parameterized.TestCase, test.TestCase):

  def test_normalize_nfkc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"ṩ".encode("utf-8"),
    ]
    actual, _ = normalize_ops.normalize_utf8_with_offsets_map(txt, "NFKC")
    self.assertAllEqual(expected, actual)
    actual, _ = normalize_ops.normalize_utf8_with_offsets_map(txt, "nfkc")
    self.assertAllEqual(expected, actual)

  def test_normalize_nfc(self):
    txt = [
        u"\u1e9b\u0323",
    ]
    expected = [
        u"\u1e9b\u0323".encode("utf-8"),
    ]
    actual, _ = normalize_ops.normalize_utf8_with_offsets_map(txt, "NFC")
    self.assertAllEqual(expected, actual)
    actual, _ = normalize_ops.normalize_utf8_with_offsets_map(txt, "nfc")
    self.assertAllEqual(expected, actual)

  def test_normalize_nfkc_batch(self):
    txt = [
        u"\u1e9b\u0323",
        u"\ufb01",
    ]
    expected = [
        b"\xe1\xb9\xa9",
        b"fi",
    ]
    actual, _ = normalize_ops.normalize_utf8_with_offsets_map(txt, u"NFKC")
    self.assertAllEqual(expected, actual)
    actual, _ = normalize_ops.normalize_utf8_with_offsets_map(txt, u"nfkc")
    self.assertAllEqual(expected, actual)

  def test_normalize_nfkc_ragged(self):
    txt = ragged_factory_ops.constant([[[u"\u1e9b\u0323 \ufb01"], []],
                                       [[u"\u1e9b\u0323", u"\ufb01"]]])
    expected = [[[u"ṩ fi".encode("utf-8")], []],
                [[u"ṩ".encode("utf-8"), b"fi"]]]
    actual, _ = normalize_ops.normalize_utf8_with_offsets_map(txt, "NFKC")
    self.assertAllEqual(expected, actual)

  def test_unaccepted_normalization_form(self):
    with self.assertRaises(errors.InvalidArgumentError):
      bomb = normalize_ops.normalize_utf8_with_offsets_map(
          ["cant readme", "wont read me"], "CANTNORMALIZEME")
      self.evaluate(bomb)


@test_util.run_all_in_graph_and_eager_modes
class FindSourceOffsetsTest(parameterized.TestCase, test.TestCase):

  def _extract_substrs(self, txt_input, start, end):
    extracted = []
    start = self.evaluate(start)
    end = self.evaluate(end)
    txt_input = txt_input.encode("utf-8")
    for i in range(start.shape[1]):
      pre_norm_start = int(start[0][i])
      pre_norm_end = int(end[0][i])
      extracted.append(txt_input[pre_norm_start:pre_norm_end])
    return extracted

  def test_one_string(self):
    txt = [
        u"株式会社ＫＡＤＯＫＡＷＡ",
    ]
    _, offsets_map = normalize_ops.normalize_utf8_with_offsets_map(txt, u"NFKC")

    # post_norm_txt = "株式会社KADOKAWA"
    post_norm_offsets_starts = [[
        0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ]]
    post_norm_offsets_ends = [[3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]

    pre_norm_offsets_starts = normalize_ops.find_source_offsets(
        offsets_map, post_norm_offsets_starts)
    pre_norm_offsets_ends = normalize_ops.find_source_offsets(
        offsets_map, post_norm_offsets_ends)
    expected_pre_norm_characters = [
        u"株", u"式", u"会", u"社", u"Ｋ", u"Ａ", u"Ｄ", u"Ｏ", u"Ｋ",
        u"Ａ", u"Ｗ", u"Ａ", u""
    ]
    self.assertAllEqual(
        self._extract_substrs(txt[0], pre_norm_offsets_starts,
                              pre_norm_offsets_ends),
        [x.encode("utf-8") for x in expected_pre_norm_characters])

  @parameterized.parameters([
      # Test one string and rank = 0 offset input
      dict(
          txt_input=["株式会社ＫＡＤＯＫＡＷＡ"],
          normalization_form="NFKC",
          post_norm_offsets=22,
          expected=36),
      # Test one string and rank = 1 offset input
      dict(
          txt_input=["株式会社ＫＡＤＯＫＡＷＡ"],
          normalization_form="NFKC",
          post_norm_offsets=[0, 1, 2],
          expected=[0, 1, 2]),
      # Test multiple strings and rank = 2 offset input
      dict(
          txt_input=[
              "株式会社",
              "ＫＡＤＯＫＡＷＡ",
          ],
          normalization_form="NFKC",
          post_norm_offsets=[[0, 1, 2], [0, 1, 2]],
          expected=[[0, 1, 2], [0, 3, 6]]),
      # Test multiple strings and rank > 2 offset input
      dict(
          txt_input=[
              ["株式会社"],
              ["ＫＡＤＯＫＡＷＡ"],
          ],
          normalization_form="NFKC",
          post_norm_offsets=[[[0, 1, 2]], [[0, 1, 2]]],
          expected=[[[0, 1, 2]], [[0, 3, 6]]]),
  ])
  def test_tensor_input(self, txt_input, normalization_form, post_norm_offsets,
                        expected):
    _, offsets_map = normalize_ops.normalize_utf8_with_offsets_map(
        txt_input, normalization_form)
    pre_norm_offsets = normalize_ops.find_source_offsets(
        offsets_map, post_norm_offsets)
    self.assertAllEqual(expected, pre_norm_offsets)

  @parameterized.parameters([
      # Test multiple strings with an empty str
      dict(
          txt_input=[
              ["株式会社"],
              [""],
              ["ＫＡＤＯＫＡＷＡ"],
          ],
          normalization_form="NFKC",
          post_norm_offsets=[[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]],
          expected=[[[0, 1, 2]], [[0, 0, 0]], [[0, 3, 6]]]),
      # Test multiple strings with an empty element
      dict(
          txt_input=[
              ["株式会社"],
              [],
              ["ＫＡＤＯＫＡＷＡ"],
          ],
          normalization_form="NFKC",
          post_norm_offsets=[[[0, 1, 2]], [[]], [[0, 1, 2]]],
          expected=[[[0, 1, 2]], [[]], [[0, 3, 6]]]),
  ])
  def test_ragged_tensor_input(self, txt_input, normalization_form,
                               post_norm_offsets, expected):
    txt_input = ragged_factory_ops.constant(txt_input)
    post_norm_offsets = ragged_factory_ops.constant(
        post_norm_offsets, dtype="int64")
    _, offsets_map = normalize_ops.normalize_utf8_with_offsets_map(
        txt_input, normalization_form)
    pre_norm_offsets = normalize_ops.find_source_offsets(
        offsets_map, post_norm_offsets)
    self.assertAllEqual(expected, pre_norm_offsets)

  def test_string_ragged_dimension_lower_than_offsets_input(self):
    txt = ragged_factory_ops.constant([
        ["株式会社"],
        [],
        ["ＫＡＤＯＫＡＷＡ"],
    ])
    _, offsets_map = normalize_ops.normalize_utf8_with_offsets_map(txt, u"NFKC")
    post_norm_offsets = ragged_factory_ops.constant(
        [[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype="int64")
    with self.assertRaises(errors.InvalidArgumentError):
      bomb = normalize_ops.find_source_offsets(offsets_map, post_norm_offsets)
      self.evaluate(bomb)

  def test_string_ragged_dimension_higher_than_offsets_input(self):
    txt = ragged_factory_ops.constant([
        ["株式会社"],
        [""],
        ["ＫＡＤＯＫＡＷＡ"],
    ])
    _, offsets_map = normalize_ops.normalize_utf8_with_offsets_map(txt, u"NFKC")
    post_norm_offsets = ragged_factory_ops.constant(
        [[[0, 1, 2]], [[]], [[0, 1, 2]]], dtype="int64")
    with self.assertRaises(errors.InvalidArgumentError):
      bomb = normalize_ops.find_source_offsets(offsets_map, post_norm_offsets)
      self.evaluate(bomb)

  def test_sliced_offsets_map_and_input_offset(self):
    txt = ragged_factory_ops.constant([
        ["株式会社"],
        [""],
        ["ＫＡＤＯＫＡＷＡ"],
    ])
    _, offsets_map = normalize_ops.normalize_utf8_with_offsets_map(txt, u"NFKC")
    post_norm_offsets = ragged_factory_ops.constant(
        [[[0, 1, 2]], [[]], [[0, 1, 2]]], dtype="int64")

    sliced_offsets_map = offsets_map[2]
    sliced_post_norm_offsets = post_norm_offsets[2]
    sliced_pre_norm_offsets = normalize_ops.find_source_offsets(
        sliced_offsets_map, sliced_post_norm_offsets)
    expected = [[0, 3, 6]]
    self.assertAllEqual(expected, sliced_pre_norm_offsets)


if __name__ == "__main__":
  test.main()
