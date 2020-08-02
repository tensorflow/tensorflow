# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for method name utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl as loader
from tensorflow.python.saved_model import method_name_updater
from tensorflow.python.util import compat

_SAVED_MODEL_PROTO = text_format.Parse("""
saved_model_schema_version: 1
meta_graphs {
  meta_info_def {
    tags: "serve"
  }
  signature_def: {
    key: "serving_default"
    value: {
      inputs: {
        key: "inputs"
        value { name: "input_node:0" }
      }
      method_name: "predict"
      outputs: {
        key: "outputs"
        value {
          dtype: DT_FLOAT
          tensor_shape {
            dim { size: -1 }
            dim { size: 100 }
          }
        }
      }
    }
  }
  signature_def: {
    key: "foo"
    value: {
      inputs: {
        key: "inputs"
        value { name: "input_node:0" }
      }
      method_name: "predict"
      outputs: {
        key: "outputs"
        value {
          dtype: DT_FLOAT
          tensor_shape { dim { size: 1 } }
        }
      }
    }
  }
}
meta_graphs {
  meta_info_def {
    tags: "serve"
    tags: "gpu"
  }
  signature_def: {
    key: "serving_default"
    value: {
      inputs: {
        key: "inputs"
        value { name: "input_node:0" }
      }
      method_name: "predict"
      outputs: {
        key: "outputs"
        value {
          dtype: DT_FLOAT
          tensor_shape {
            dim { size: -1 }
          }
        }
      }
    }
  }
  signature_def: {
    key: "bar"
    value: {
      inputs: {
        key: "inputs"
        value { name: "input_node:0" }
      }
      method_name: "predict"
      outputs: {
        key: "outputs"
        value {
          dtype: DT_FLOAT
          tensor_shape { dim { size: 1 } }
        }
      }
    }
  }
}
""", saved_model_pb2.SavedModel())


class MethodNameUpdaterTest(test.TestCase):

  def setUp(self):
    super(MethodNameUpdaterTest, self).setUp()
    self._saved_model_path = tempfile.mkdtemp(prefix=test.get_temp_dir())

  def testBasic(self):
    path = os.path.join(
        compat.as_bytes(self._saved_model_path),
        compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    file_io.write_string_to_file(
        path, _SAVED_MODEL_PROTO.SerializeToString(deterministic=True))

    updater = method_name_updater.MethodNameUpdater(self._saved_model_path)
    updater.replace_method_name(
        signature_key="serving_default", method_name="classify")
    updater.save()

    actual = loader.parse_saved_model(self._saved_model_path)
    self.assertProtoEquals(
        actual,
        text_format.Parse(
            """
        saved_model_schema_version: 1
        meta_graphs {
          meta_info_def {
            tags: "serve"
          }
          signature_def: {
            key: "serving_default"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "classify"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape {
                    dim { size: -1 }
                    dim { size: 100 }
                  }
                }
              }
            }
          }
          signature_def: {
            key: "foo"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "predict"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        meta_graphs {
          meta_info_def {
            tags: "serve"
            tags: "gpu"
          }
          signature_def: {
            key: "serving_default"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "classify"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape {
                    dim { size: -1 }
                  }
                }
              }
            }
          }
          signature_def: {
            key: "bar"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "predict"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape { dim { size: 1 } }
                }
              }
            }
          }
        }
    """, saved_model_pb2.SavedModel()))

  def testTextFormatAndNewExportDir(self):
    path = os.path.join(
        compat.as_bytes(self._saved_model_path),
        compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
    file_io.write_string_to_file(path, str(_SAVED_MODEL_PROTO))

    updater = method_name_updater.MethodNameUpdater(self._saved_model_path)
    updater.replace_method_name(
        signature_key="foo", method_name="regress", tags="serve")
    updater.replace_method_name(
        signature_key="bar", method_name="classify", tags=["gpu", "serve"])

    new_export_dir = tempfile.mkdtemp(prefix=test.get_temp_dir())
    updater.save(new_export_dir)

    self.assertTrue(
        file_io.file_exists(
            os.path.join(
                compat.as_bytes(new_export_dir),
                compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))))
    actual = loader.parse_saved_model(new_export_dir)
    self.assertProtoEquals(
        actual,
        text_format.Parse(
            """
        saved_model_schema_version: 1
        meta_graphs {
          meta_info_def {
            tags: "serve"
          }
          signature_def: {
            key: "serving_default"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "predict"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape {
                    dim { size: -1 }
                    dim { size: 100 }
                  }
                }
              }
            }
          }
          signature_def: {
            key: "foo"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "regress"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        meta_graphs {
          meta_info_def {
            tags: "serve"
            tags: "gpu"
          }
          signature_def: {
            key: "serving_default"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "predict"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape {
                    dim { size: -1 }
                  }
                }
              }
            }
          }
          signature_def: {
            key: "bar"
            value: {
              inputs: {
                key: "inputs"
                value { name: "input_node:0" }
              }
              method_name: "classify"
              outputs: {
                key: "outputs"
                value {
                  dtype: DT_FLOAT
                  tensor_shape { dim { size: 1 } }
                }
              }
            }
          }
        }
    """, saved_model_pb2.SavedModel()))

  def testExceptions(self):
    with self.assertRaises(IOError):
      updater = method_name_updater.MethodNameUpdater(
          tempfile.mkdtemp(prefix=test.get_temp_dir()))

    path = os.path.join(
        compat.as_bytes(self._saved_model_path),
        compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    file_io.write_string_to_file(
        path, _SAVED_MODEL_PROTO.SerializeToString(deterministic=True))
    updater = method_name_updater.MethodNameUpdater(self._saved_model_path)

    with self.assertRaisesRegex(ValueError, "signature_key must be defined"):
      updater.replace_method_name(
          signature_key=None, method_name="classify")

    with self.assertRaisesRegex(ValueError, "method_name must be defined"):
      updater.replace_method_name(
          signature_key="foobar", method_name="")

    with self.assertRaisesRegex(
        ValueError,
        r"MetaGraphDef associated with tags \['gpu'\] could not be found"):
      updater.replace_method_name(
          signature_key="bar", method_name="classify", tags=["gpu"])

    with self.assertRaisesRegex(
        ValueError, r"MetaGraphDef associated with tags \['serve'\] does not "
                    r"have a signature_def with key: baz"):
      updater.replace_method_name(
          signature_key="baz", method_name="classify", tags=["serve"])

if __name__ == "__main__":
  test.main()
