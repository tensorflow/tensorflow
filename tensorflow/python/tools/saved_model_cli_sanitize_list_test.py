# tensorflow/python/tools/saved_model_cli_sanitize_list_test.py
# Copyright 2025 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0
# pylint: disable=line-too-long

"""Unit tests for trim & validation helpers in saved_model_cli."""

from absl.testing import parameterized
from tensorflow.python.platform import test
from tensorflow.python.tools import saved_model_cli as smcli


class SanitizeNonEmptyStrListTest(test.TestCase, parameterized.TestCase):

  def test_trims_and_drops_empty_and_none(self):
    self.assertEqual(
        smcli._sanitize_nonempty_str_list(
            [' a ', '', '\t', None, 'b '], 'field'
        ),
        ['a', 'b'],
    )

  def test_raises_on_all_empty_like_inputs(self):
    with self.assertRaisesRegex(ValueError, 'field.*at least one non-empty'):
      smcli._sanitize_nonempty_str_list(['  ', '\n', '', None], 'field')


class DenylistParsingTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('OpA, OpB, , ,,OpC', {'OpA', 'OpB', 'OpC'}),
      ('  , , ', set()),
      ('', set()),
      (' ReadFile ,WriteFile , PrintV2', {'ReadFile', 'WriteFile', 'PrintV2'}),
  )
  def test_get_op_denylist_set_trims_and_ignores_empties(self, raw, expected):
    self.assertEqual(smcli._get_op_denylist_set(raw), expected)


if __name__ == '__main__':
  test.main()
