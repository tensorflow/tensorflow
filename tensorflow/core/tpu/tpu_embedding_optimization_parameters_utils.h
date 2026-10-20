/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OPTIMIZATION_PARAMETERS_UTILS_H_
#define TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OPTIMIZATION_PARAMETERS_UTILS_H_

#include "absl/status/status.h"
#include "xla/tpu/protobuf/tpu_embedding_optimization_parameters_utils.h"  // IWYU pragma: export
#include "tensorflow/core/framework/op_def_builder.h"

namespace tensorflow {
namespace tpu {

// Generic shape function for per-optimization-algorithm load ops.
class LoadOpShapeFunction {
 public:
  // Computes resulting shape and does parameter checking.
  absl::Status operator()(shape_inference::InferenceContext *c) const;
};

// Generic shape function for per-optimization-algorithm retrieve ops.
class RetrieveOpShapeFunction {
 public:
  // Computes resulting shape and does parameter checking.
  absl::Status operator()(shape_inference::InferenceContext *c) const;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_EMBEDDING_OPTIMIZATION_PARAMETERS_UTILS_H_
