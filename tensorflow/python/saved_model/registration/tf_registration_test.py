# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests TensorFlow registered objects.

New or deleted registrations must be approved by the Saved Model team.
"""
import os

import tensorflow as tf

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model.registration import test_util

SERIALIZABLE_ALLOWLIST = os.path.join(resource_loader.get_data_files_path(),
                                      "tf_serializable_allowlist.txt")
CHECKPOINT_SAVER_ALLOWLIST = os.path.join(resource_loader.get_data_files_path(),
                                          "tf_checkpoint_saver_allowlist.txt")


@registration.register_tf_serializable()
class NotInAllowlistExample(tf.Variable):
  pass


registration.register_tf_checkpoint_saver(
    name="TestDummySaver",
    predicate=lambda: False,
    save_fn=lambda: None,
    restore_fn=lambda: None)

registration.register_tf_checkpoint_saver(
    name="NotInAllowlistExample",
    predicate=lambda: False,
    save_fn=lambda: None,
    restore_fn=lambda: None)


class AllowlistTest(test.TestCase):

  def _test_allowlist(self, allowlist_file, registrations):
    allowlist = set([
        s.strip()
        for s in file_io.read_file_to_string(allowlist_file).splitlines()
        if s.strip() and not s.startswith("#")
    ])
    registered_names = set(registrations)

    missing_from_allowlist = registered_names - allowlist
    self.assertIn("tf.NotInAllowlistExample", missing_from_allowlist)
    missing_from_allowlist.remove("tf.NotInAllowlistExample")

    if missing_from_allowlist:
      msg = ("[NEEDS ATTENTION] Registered names found that were not added to "
             "the allowlist. Add the following names to the list:\n\t" +
             "\n\t".join(missing_from_allowlist))
    else:
      msg = "[OK] All registered names have been added to the allowlist.  ✓"

    msg += "\n\n"

    missing_registered_names = allowlist - registered_names
    if missing_registered_names:
      msg += ("[NEEDS ATTENTION] Some names were found in the allowlist that "
              "are not registered in TensorFlow. This could mean that a "
              "registration was removed from the codebase. If this was "
              "intended, please remove the following from the allowlist:\n\t" +
              "\n\t".join(missing_registered_names))
    else:
      msg += ("[OK] All allowlisted names are registered in the Tensorflow "
              "library. ✓")

    if missing_from_allowlist or missing_registered_names:
      raise AssertionError(
          "Error found in the registration allowlist.\nPlease update the "
          "allowlist at .../tensorflow/python/saved_model/registration/"
          f"{os.path.basename(allowlist_file)}.\n\n" + msg +
          "\n\nAfter making changes, request approval from "
          " tf-saved-model-owners@.")

  def test_checkpoint_savers(self):
    self._test_allowlist(CHECKPOINT_SAVER_ALLOWLIST,
                         test_util.get_all_registered_checkpoint_savers())

  def test_serializables(self):
    self._test_allowlist(SERIALIZABLE_ALLOWLIST,
                         test_util.get_all_registered_serializables())


if __name__ == "__main__":
  test.main()
