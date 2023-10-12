/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SPLIT_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SPLIT_UTILS_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace grappler {
namespace split_utils {

// Return value of `SplitFunction`, which is described below.
struct SplitResults {
  FunctionDef first_function;
  FunctionDef second_function;
  std::vector<DataType> first_function_output_types;
};

// Splits a FunctionDef into two FunctionDefs, called `first` and `second`, such
// that calling `function(*args)` is equivalent to calling
// `second(first(*args))`. The set `nodes_in_first_function` specifies nodes
// that are copied to `first`, and the other nodes are copied to `second`. Any
// edges from `first` to `second` will be represented by an output of `first`
// and a corresponding input of `second`. The caller must pass
// `nodes_in_first_function` such that there will not be any edges from `second`
// to `first`.
//
// For example, if you have the following function (using Python syntax):
//
//     def f(x):
//       y = tf.math.add(x, 1., name='add')
//       return tf.multiply(y, 2, name='mul')
//
// Calling SplitFunction(f, {'add'}) results in:
//
//     def first_function(x):
//       return tf.math.add(x, 1., name='add')
//     def second_function(y):
//       return tf.multiply(y, 2, name='mul')
//
// The `num_captured_inputs` argument controls which arguments of `function`
// will be arguments of `second`. If it is zero, the only arguments of `second`
// are the outputs of `first`. If it is above zero, the last
// `num_caputured_inputs` arguments of `function` will also be arguments of
// `second`.
//
// Splitting functions in certain cases is unimplemented, in which case an
// Unimplemented status will be returned. Grappler passes must gracefully handle
// Unimplemented statuses without returning the error to its caller.
StatusOr<SplitResults> SplitFunction(
    const FunctionDef& function,
    const absl::flat_hash_set<absl::string_view>& nodes_in_first_function,
    int64_t num_captured_inputs, const FunctionLibraryDefinition& library);

}  // namespace split_utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SPLIT_UTILS_H_
