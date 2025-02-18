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
#include <utility>
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
// Each sub-config is responsible for one feature identified with feature_name.
// FastParseExampleConfig can't have two sub-configs with the same feature_name.
// dtype identifies the type of output vector and the kind of Feature expected
// in Example.
struct FastParseExampleConfig {
  struct Dense {
    Dense(absl::string_view feature_name, DataType dtype,
          PartialTensorShape shape, Tensor default_value, bool variable_length,
          std::size_t elements_per_stride)
        : feature_name(feature_name),  // TODO(mrry): Switch to preallocated
                                       // tstring when this is available.
          dtype(dtype),
          shape(std::move(shape)),
          default_value(std::move(default_value)),
          variable_length(variable_length),
          elements_per_stride(elements_per_stride) {}
    Dense() = default;

    tstring feature_name;
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
    Sparse(absl::string_view feature_name, DataType dtype)
        : feature_name(feature_name),  // TODO(mrry): Switch to preallocated
                                       // tstring when this is available.
          dtype(dtype) {}
    Sparse() = default;

    tstring feature_name;
    DataType dtype;
  };

  struct Ragged {
    Ragged(absl::string_view feature_name, DataType dtype,
           DataType splits_dtype)
        : feature_name(feature_name),  // TODO(mrry): Switch to preallocated
                                       // tstring when this is available.
          dtype(dtype),
          splits_dtype(splits_dtype) {}
    Ragged() = default;

    tstring feature_name;
    DataType dtype;
    DataType splits_dtype;
  };

  std::vector<Dense> dense;
  std::vector<Sparse> sparse;
  std::vector<Ragged> ragged;

  // If `true`, `Result::feature_stats` will contain one
  // `PerExampleFeatureStats` for each serialized example in the input.
  bool collect_feature_stats = false;
};

// Statistics about the features in each example passed to
// `FastParse[Single]Example()`.
//
// TODO(b/111553342): The gathered statistics currently have two limitations:
// * Feature names that appear more than once will be counted multiple times.
// * The feature values count only represents the counts for features that were
//   requested in the `FastParseExampleConfig`.
// These could be addressed with additional work at runtime.
struct PerExampleFeatureStats {
  // The number of feature names in an example.
  size_t features_count = 0;

  // The sum of the number of values in each feature that is parsed.
  size_t feature_values_count = 0;
};

// This is exactly the output of TF's ParseExample Op.
// Documentation is available in: tensorflow/core/ops/parsing_ops.cc
struct Result {
  std::vector<Tensor> sparse_indices;
  std::vector<Tensor> sparse_values;
  std::vector<Tensor> sparse_shapes;
  std::vector<Tensor> dense_values;
  std::vector<Tensor> ragged_values;
  std::vector<Tensor> ragged_splits;
  std::vector<Tensor> ragged_outer_splits;  // For SequenceExamples

  // This vector will be populated with one element per example if
  // `FastParseExampleConfig::collect_feature_stats` is set to `true`.
  std::vector<PerExampleFeatureStats> feature_stats;
};

// Parses a batch of serialized Example protos and converts them into result
// according to given config.
// Given example names have to either be empty or the same size as serialized.
// example_names are used only for error messages.
absl::Status FastParseExample(const FastParseExampleConfig& config,
                              absl::Span<const tstring> serialized,
                              absl::Span<const tstring> example_names,
                              thread::ThreadPool* thread_pool, Result* result);

// TODO(mrry): Move the hash table construction into the config object.
typedef FastParseExampleConfig FastParseSingleExampleConfig;

absl::Status FastParseSingleExample(const FastParseSingleExampleConfig& config,
                                    absl::string_view serialized,
                                    Result* result);

// Parses a batch of serialized SequenceExample protos and converts them into
// result according to given config.
// Given example names have to either be empty or the same size as serialized.
// example_names are used only for error messages.
// (If batch=true, then this parses a single SequenceExample.)
absl::Status FastParseSequenceExample(
    const example::FastParseExampleConfig& context_config,
    const example::FastParseExampleConfig& sequence_config,
    absl::Span<const tstring> serialized,
    absl::Span<const tstring> example_names, thread::ThreadPool* thread_pool,
    example::Result* context_result, example::Result* sequence_result,
    std::vector<Tensor>* dense_feature_lengths, bool is_batch = true);

// This function parses serialized Example and populates given example.
// It uses the same specialized parser as FastParseExample which is efficient.
// But then constructs Example which is relatively slow.
// It is exported here as a convenient API to test parser part separately.
bool TestFastParse(const string& serialized, Example* example);

}  // namespace example
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_EXAMPLE_PROTO_FAST_PARSING_H_
