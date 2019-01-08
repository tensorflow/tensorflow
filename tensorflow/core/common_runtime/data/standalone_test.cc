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

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/data/standalone.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace standalone {
namespace {

constexpr const char* const kGraphProto = R"proto(
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
      value { list { shape { unknown_rank: true } } }
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
      value { func { name: "Dataset_map_<lambda>_10" } }
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
  library {
    function {
      signature {
        name: "Dataset_map_<lambda>_10"
        input_arg { name: "arg0" type: DT_INT64 }
        output_arg { name: "mul" type: DT_INT64 }
        description: "Wrapper for passing nested structures to and from tf.data functions."
      }
      node_def {
        name: "mul_0"
        op: "Mul"
        input: "arg0"
        input: "arg0"
        attr {
          key: "T"
          value { type: DT_INT64 }
        }
      }
      ret { key: "mul" value: "mul_0:z:0" }
    }
  }
  versions { producer: 27 min_consumer: 12 }
)proto";

TEST(Scalar, Standalone) {
  GraphDef graph_def;
  protobuf::TextFormat::ParseFromString(kGraphProto, &graph_def);
  struct TestCase {
    string fetch_node;
    std::vector<int64> expected_outputs;
  };
  auto test_cases = {
      TestCase{"RangeDataset/_3", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
      TestCase{"MapDataset/_4", {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}},
  };
  for (auto test_case : test_cases) {
    std::unique_ptr<Dataset> dataset;
    auto s = Dataset::FromGraph({}, graph_def, test_case.fetch_node, &dataset);
    TF_EXPECT_OK(s);
    std::unique_ptr<Iterator> iterator;
    s = dataset->MakeIterator(&iterator);
    TF_EXPECT_OK(s);
    bool end_of_input = false;
    for (int num_outputs = 0; !end_of_input; ++num_outputs) {
      std::vector<tensorflow::Tensor> outputs;
      s = iterator->GetNext(&outputs, &end_of_input);
      TF_EXPECT_OK(s);
      if (!end_of_input) {
        EXPECT_EQ(outputs[0].scalar<int64>()(),
                  test_case.expected_outputs[num_outputs]);
      } else {
        EXPECT_EQ(test_case.expected_outputs.size(), num_outputs);
      }
    }
  }
}

}  // namespace
}  // namespace standalone
}  // namespace data
}  // namespace tensorflow
