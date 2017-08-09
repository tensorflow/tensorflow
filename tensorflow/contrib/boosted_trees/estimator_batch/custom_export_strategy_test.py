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
"""Tests for the conversion code from GTFlow format to Chauffeur."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import text_format
from tensorflow.contrib.boosted_trees.estimator_batch import custom_export_strategy
from tensorflow.contrib.boosted_trees.proto import tree_config_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ConvertModelTest(test_util.TensorFlowTestCase):

  def testConvertModel(self):
    dtec_str = """
    trees {
      nodes {
        leaf {
          vector {
            value: -1
          }
        }
      }
    }
    trees {
      nodes {
        dense_float_binary_split {
          feature_column: 0
          threshold: 1740.0
          left_id: 1
          right_id: 2
        }
        node_metadata {
          gain: 500
        }
      }
      nodes {
        leaf {
          vector {
            value: 0.6
          }
        }
      }
      nodes {
        sparse_float_binary_split_default_left {
          split {
            feature_column: 0
            threshold: 1500.0
            left_id: 3
            right_id: 4
          }
        }
        node_metadata {
          gain: 500
        }
      }
      nodes {
        categorical_id_binary_split {
          feature_column: 0
          feature_id: 5
          left_id: 5
          right_id: 6
        }
        node_metadata {
          gain: 500
        }
      }
      nodes {
        leaf {
          vector {
            value: 0.8
          }
        }
      }
      nodes {
        leaf {
          vector {
            value: 0.5
          }
        }
      }
      nodes {
        leaf {
          vector {
            value: 0.3
          }
        }
      }
    }
    tree_weights: 1.0
    tree_weights: 0.1
    """
    dtec = tree_config_pb2.DecisionTreeEnsembleConfig()
    text_format.Merge(dtec_str, dtec)
    # The feature columns in the order they were added.
    feature_columns = ["feature_b", "feature_a", "feature_d"]
    out = custom_export_strategy.convert_to_universal_format(
        dtec, feature_columns, 1, 1,
        1)
    expected_tree = """
    features { key: "feature_a" }
    features { key: "feature_b" }
    features { key: "feature_d" }
    model {
      ensemble {
        summation_combination_technique {
        }
        members {
          submodel {
            decision_tree {
              nodes {
                node_id {
                }
                leaf {
                  vector {
                    value {
                      float_value: -1.0
                    }
                  }
                }
              }
            }
          }
          submodel_id {
          }
        }
        members {
          submodel {
            decision_tree {
              nodes {
                node_id {
                }
                binary_node {
                  left_child_id {
                    value: 1
                  }
                  right_child_id {
                    value: 2
                  }
                  inequality_left_child_test {
                    feature_id {
                      id {
                        value: "feature_b"
                      }
                    }
                    threshold {
                      float_value: 1740.0
                    }
                  }
                }
              }

              nodes {
                node_id {
                  value: 1
                }
                leaf {
                  vector {
                    value {
                      float_value: 0.06
                    }
                  }
                }
              }
              nodes {
                node_id {
                  value: 2
                }
                binary_node {
                  left_child_id {
                    value: 3
                  }
                  right_child_id {
                    value: 4
                  }
                  inequality_left_child_test {
                    feature_id {
                      id {
                        value: "feature_a"
                      }
                    }
                    threshold {
                      float_value: 1500.0
                    }
                  }
                }
              }
              nodes {
                node_id {
                  value: 3
                }
                binary_node {
                  left_child_id {
                    value: 5
                  }
                  right_child_id {
                    value: 6
                  }
                  default_direction: RIGHT
                  custom_left_child_test {
                    [type.googleapis.com/tensorflow.decision_trees.MatchingValuesTest] {
                      feature_id {
                        id {
                          value: "feature_d"
                        }
                      }
                      value {
                        int64_value: 5
                      }
                    }
                  }
                }
              }
              nodes {
                node_id {
                  value: 4
                }
                leaf {
                  vector {
                    value {
                      float_value: 0.08
                    }
                  }
                }
              }
              nodes {
                node_id {
                  value: 5
                }
                leaf {
                  vector {
                    value {
                      float_value: 0.05
                    }
                  }
                }
              }
              nodes {
                node_id {
                  value: 6
                }
                leaf {
                  vector {
                    value {
                      float_value: 0.03
                    }
                  }
                }
              }
            }
          }
          submodel_id {
            value: 1
          }
        }
      }
    }"""
    self.assertProtoEquals(expected_tree, out)


if __name__ == "__main__":
  googletest.main()
