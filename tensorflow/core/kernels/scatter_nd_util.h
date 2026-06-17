/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_ND_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_ND_UTIL_H_

#include "xla/tsl/util/env_var.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

// Validates the input shapes for the ScatterNdUpdateOp<scatter_nd_op::UpdateOp>
absl::Status ValidateScatterNdUpdateShape(const TensorShape& params_shape,
                                          const TensorShape& indices_shape,
                                          const TensorShape& updates_shape);

inline bool DisableScatterOpDeterminism() {
  static bool cached_disable = [] {
    bool disable = false;
    // When determinism is enabled, the kernels for various scatter ops like
    // ScatterNdAdd will still use the faster non-deterministic versions if this
    // environmental variable is true. This is useful if the user is certain the
    // scatter inputs don't have duplicate indices (in which cases scatter ops
    // are always deterministic), since the deterministic implementations are
    // currently slow.
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_DISABLE_SCATTER_OP_DETERMINISM",
                                        /*default_val=*/false, &disable));
    return disable;
  }();
  return cached_disable;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_ND_UTIL_H_
