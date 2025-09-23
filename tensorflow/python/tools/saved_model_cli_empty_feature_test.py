# pylint: disable=line-too-long
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import saved_model_cli_test_stubs as cli_test_stubs  # pylint: disable=g-import-not-at-top


class SavedModelCliExampleEmptyFeatureTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cli_test_stubs.install_saved_model_cli_stubs()
    cls.saved_model_cli = importlib.import_module('tensorflow.python.tools.saved_model_cli')

  def test_create_example_string_rejects_empty_feature_list(self):
    with self.assertRaisesRegex(ValueError, 'must contain at least one value'):
      self.saved_model_cli._create_example_string({'ids': []})

  def test_create_example_string_accepts_int_values(self):
    payload = self.saved_model_cli._create_example_string({'ids': [1, 2, 3]})
    self.assertIsInstance(payload, bytes)


if __name__ == '__main__':
  unittest.main()
