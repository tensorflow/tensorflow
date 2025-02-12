/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/data/standalone.h"

#include <memory>
#include <optional>
#include <vector>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace standalone {
namespace {

constexpr const char* const kRangeGraphProto = R"pb(
  node {
    name: "Const/_0"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 0
        }
      }
    }
  }
  node {
    name: "Const/_1"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 10
        }
      }
    }
  }
  node {
    name: "Const/_2"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 1
        }
      }
    }
  }
  node {
    name: "RangeDataset/_3"
    op: "RangeDataset"
    input: "Const/_0"
    input: "Const/_1"
    input: "Const/_2"
    attr {
      key: "output_shapes"
      value { list { shape {} } }
    }
    attr {
      key: "output_types"
      value { list { type: DT_INT64 } }
    }
  }
  node {
    name: "dataset"
    op: "_Retval"
    input: "RangeDataset/_3"
    attr {
      key: "T"
      value { type: DT_VARIANT }
    }
    attr {
      key: "index"
      value { i: 0 }
    }
  }
  library {}
  versions { producer: 96 }
)pb";

// range(10).map(lambda x: x*x)
constexpr const char* const kMapGraphProto = R"pb(
  node {
    name: "Const/_0"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 0
        }
      }
    }
  }
  node {
    name: "Const/_1"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 10
        }
      }
    }
  }
  node {
    name: "Const/_2"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 1
        }
      }
    }
  }
  node {
    name: "RangeDataset/_3"
    op: "RangeDataset"
    input: "Const/_0"
    input: "Const/_1"
    input: "Const/_2"
    attr {
      key: "output_shapes"
      value { list { shape {} } }
    }
    attr {
      key: "output_types"
      value { list { type: DT_INT64 } }
    }
  }
  node {
    name: "MapDataset/_4"
    op: "MapDataset"
    input: "RangeDataset/_3"
    attr {
      key: "Targuments"
      value { list {} }
    }
    attr {
      key: "f"
      value { func { name: "__inference_Dataset_map_<lambda>_67" } }
    }
    attr {
      key: "output_shapes"
      value { list { shape {} } }
    }
    attr {
      key: "output_types"
      value { list { type: DT_INT64 } }
    }
    attr {
      key: "preserve_cardinality"
      value { b: false }
    }
    attr {
      key: "use_inter_op_parallelism"
      value { b: true }
    }
  }
  node {
    name: "dataset"
    op: "_Retval"
    input: "MapDataset/_4"
    attr {
      key: "T"
      value { type: DT_VARIANT }
    }
    attr {
      key: "index"
      value { i: 0 }
    }
  }
  library {
    function {
      signature {
        name: "__inference_Dataset_map_<lambda>_67"
        input_arg { name: "args_0" type: DT_INT64 }
        output_arg { name: "identity" type: DT_INT64 }
      }
      node_def {
        name: "mul"
        op: "Mul"
        input: "args_0"
        input: "args_0"
        attr {
          key: "T"
          value { type: DT_INT64 }
        }
      }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "mul:z:0"
        attr {
          key: "T"
          value { type: DT_INT64 }
        }
      }
      ret { key: "identity" value: "Identity:output:0" }
      arg_attr {
        key: 0
        value {
          attr {
            key: "_user_specified_name"
            value { s: "args_0" }
          }
        }
      }
    }
  }
  versions { producer: 96 min_consumer: 12 }
)pb";

// ds = tf.data.Dataset.range(10).map(lambda x: x*x)
// options = tf.data.Options()
// options.autotune.enabled = False
// ds = ds.with_options(options)
constexpr const char* const kMapGraphNoAutotuneProto = R"pb(
  node {
    name: "Const/_0"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 0
        }
      }
    }
  }
  node {
    name: "Const/_1"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 10
        }
      }
    }
  }
  node {
    name: "Const/_2"
    op: "Const"
    attr {
      key: "dtype"
      value { type: DT_INT64 }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {}
          int64_val: 1
        }
      }
    }
  }
  node {
    name: "RangeDataset/_3"
    op: "RangeDataset"
    input: "Const/_0"
    input: "Const/_1"
    input: "Const/_2"
    attr {
      key: "metadata"
      value { s: "\n\017RangeDataset:13" }
    }
    attr {
      key: "output_shapes"
      value { list { shape {} } }
    }
    attr {
      key: "output_types"
      value { list { type: DT_INT64 } }
    }
    attr {
      key: "replicate_on_split"
      value { b: false }
    }
    experimental_type {
      type_id: TFT_PRODUCT
      args {
        type_id: TFT_DATASET
        args {
          type_id: TFT_PRODUCT
          args {
            type_id: TFT_TENSOR
            args { type_id: TFT_INT64 }
          }
        }
      }
    }
  }
  node {
    name: "MapDataset/_4"
    op: "MapDataset"
    input: "RangeDataset/_3"
    attr {
      key: "Targuments"
      value { list {} }
    }
    attr {
      key: "f"
      value { func { name: "__inference_Dataset_map_lambda_74" } }
    }
    attr {
      key: "metadata"
      value { s: "\n\rMapDataset:14" }
    }
    attr {
      key: "output_shapes"
      value { list { shape {} } }
    }
    attr {
      key: "output_types"
      value { list { type: DT_INT64 } }
    }
    attr {
      key: "preserve_cardinality"
      value { b: true }
    }
    attr {
      key: "use_inter_op_parallelism"
      value { b: true }
    }
    experimental_type {
      type_id: TFT_PRODUCT
      args {
        type_id: TFT_DATASET
        args {
          type_id: TFT_PRODUCT
          args {
            type_id: TFT_TENSOR
            args { type_id: TFT_INT64 }
          }
        }
      }
    }
  }
  node {
    name: "OptionsDataset/_5"
    op: "OptionsDataset"
    input: "MapDataset/_4"
    attr {
      key: "metadata"
      value { s: "\n\021OptionsDataset:15" }
    }
    attr {
      key: "output_shapes"
      value { list { shape {} } }
    }
    attr {
      key: "output_types"
      value { list { type: DT_INT64 } }
    }
    attr {
      key: "serialized_options"
      value { s: "\022\000\032\003\240\001\000*\000:\002\010\000" }
    }
    experimental_type {
      type_id: TFT_PRODUCT
      args {
        type_id: TFT_DATASET
        args {
          type_id: TFT_PRODUCT
          args {
            type_id: TFT_TENSOR
            args { type_id: TFT_INT64 }
          }
        }
      }
    }
  }
  node {
    name: "dataset"
    op: "_Retval"
    input: "OptionsDataset/_5"
    attr {
      key: "T"
      value { type: DT_VARIANT }
    }
    attr {
      key: "index"
      value { i: 0 }
    }
  }
  library {
    function {
      signature {
        name: "__inference_Dataset_map_lambda_74"
        input_arg { name: "args_0" type: DT_INT64 }
        output_arg { name: "identity" type: DT_INT64 }
      }
      node_def {
        name: "mul"
        op: "Mul"
        input: "args_0"
        input: "args_0"
        attr {
          key: "T"
          value { type: DT_INT64 }
        }
      }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "mul:z:0"
        attr {
          key: "T"
          value { type: DT_INT64 }
        }
      }
      ret { key: "identity" value: "Identity:output:0" }
      attr {
        key: "_construction_context"
        value { s: "kEagerRuntime" }
      }
      attr {
        key: "_tf_data_function"
        value { b: true }
      }
      arg_attr {
        key: 0
        value {
          attr {
            key: "_output_shapes"
            value { list { shape {} } }
          }
          attr {
            key: "_user_specified_name"
            value { s: "args_0" }
          }
        }
      }
    }
  }
  versions { producer: 1594 }
)pb";

TEST(Scalar, Standalone) {
  struct TestCase {
    string graph_string;
    std::vector<int64_t> expected_outputs;
  };
  auto test_cases = {
      TestCase{kRangeGraphProto, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
      TestCase{kMapGraphProto, {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}},
  };
  for (auto test_case : test_cases) {
    GraphDef graph_def;
    protobuf::TextFormat::ParseFromString(test_case.graph_string, &graph_def);
    std::unique_ptr<Dataset> dataset;
    TF_EXPECT_OK(Dataset::FromGraph({}, graph_def, &dataset));
    std::unique_ptr<Iterator> iterator;
    TF_EXPECT_OK(dataset->MakeIterator(&iterator));
    EXPECT_DOUBLE_EQ(iterator->model()->ComputeSnapshotProcessingTimeNsec(), 0);

    bool end_of_input = false;
    for (int num_outputs = 0; !end_of_input; ++num_outputs) {
      std::vector<tensorflow::Tensor> outputs;
      TF_EXPECT_OK(iterator->GetNext(&outputs, &end_of_input));
      if (!end_of_input) {
        EXPECT_EQ(outputs[0].scalar<int64_t>()(),
                  test_case.expected_outputs[num_outputs]);
      } else {
        EXPECT_EQ(test_case.expected_outputs.size(), num_outputs);
      }
    }

    // Wait for an optimization round in the pipeline model.
    absl::SleepFor(absl::Seconds(1));
    EXPECT_GT(iterator->model()->ComputeSnapshotProcessingTimeNsec(), 0);
  }
}

TEST(NoAutotune, Standalone) {
  std::vector<int64_t> expected_outputs({0, 1, 4, 9, 16, 25, 36, 49, 64, 81});
  GraphDef graph_def;
  protobuf::TextFormat::ParseFromString(kMapGraphNoAutotuneProto, &graph_def);
  std::unique_ptr<Dataset> dataset;
  TF_EXPECT_OK(Dataset::FromGraph({}, graph_def, &dataset));
  std::unique_ptr<Iterator> iterator;
  TF_EXPECT_OK(dataset->MakeIterator(&iterator));
  EXPECT_EQ(iterator->model(), nullptr);

  bool end_of_input = false;
  for (int num_outputs = 0; !end_of_input; ++num_outputs) {
    std::vector<tensorflow::Tensor> outputs;
    TF_EXPECT_OK(iterator->GetNext(&outputs, &end_of_input));
    if (!end_of_input) {
      EXPECT_EQ(outputs[0].scalar<int64_t>()(), expected_outputs[num_outputs]);
    } else {
      EXPECT_EQ(expected_outputs.size(), num_outputs);
    }
  }

  // Wait for an optimization round in the pipeline model.
  absl::SleepFor(absl::Seconds(1));
  // Model should not be created.
  EXPECT_EQ(iterator->model(), nullptr);
}

}  // namespace
}  // namespace standalone
}  // namespace data
}  // namespace tensorflow
