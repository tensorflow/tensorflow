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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_RANDOM_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_RANDOM_H_

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
// Builds an array filled with values sampled from a truncated normal
// distribution such that no values are greater than two or less than negative
// two.
//
// The "uniform" parameter must be an array of random numbers distributed in
// (0,1).
xla::StatusOr<xla::XlaOp> TruncatedNormal(DataType dtype,
                                          const xla::XlaOp& uniform,
                                          xla::XlaBuilder* builder);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_RANDOM_H_
