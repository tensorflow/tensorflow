/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_FAST_PARSING_H_
#define TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_FAST_PARSING_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
namespace example {

// FastParseExampleConfig defines how to parse features in Example.
// Each sub-config is responsible for one feature identified with feautre_name.
// FastParseExampleConfig can't have two sub-configs with the same feature_name.
// dtype identifies the type of output vector and the kind of Feature expected
// in Example.
struct FastParseExampleConfig {
  struct Dense {
    string feature_name;
    DataType dtype;
    // These 2 fields correspond exactly to dense_shapes and dense_defaults in
    // ParseExample op.
    // Documentation is available in: tensorflow/core/ops/parsing_ops.cc
    PartialTensorShape shape;
    Tensor default_value;
    bool variable_length;
    std::size_t elements_per_stride;
  };

  struct Sparse {
    string feature_name;
    DataType dtype;
  };

  std::vector<Dense> dense;
  std::vector<Sparse> sparse;
};

// This is exactly the output of TF's ParseExample Op.
// Documentation is available in: tensorflow/core/ops/parsing_ops.cc
struct Result {
  std::vector<Tensor> sparse_indices;
  std::vector<Tensor> sparse_values;
  std::vector<Tensor> sparse_shapes;
  std::vector<Tensor> dense_values;
};

// Parses a batch of serialized Example protos and converts them into result
// according to given config.
// Given example names have to either be empty or the same size as serialized.
// example_names are used only for error messages.
Status FastParseExample(const FastParseExampleConfig& config,
                        gtl::ArraySlice<string> serialized,
                        gtl::ArraySlice<string> example_names,
                        thread::ThreadPool* thread_pool, Result* result);

// TODO(mrry): Move the hash table construction into the config object.
typedef FastParseExampleConfig FastParseSingleExampleConfig;

Status FastParseSingleExample(const FastParseSingleExampleConfig& config,
                              const string& serialized, Result* result);

// This function parses serialized Example and populates given example.
// It uses the same specialized parser as FastParseExample which is efficient.
// But then constructs Example which is relatively slow.
// It is exported here as a convenient API to test parser part separately.
bool TestFastParse(const string& serialized, Example* example);

}  // namespace example
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_FAST_PARSING_H_
