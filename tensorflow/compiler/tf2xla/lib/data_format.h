/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_DATA_FORMAT_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_DATA_FORMAT_H_

#include "xla/client/xla_builder.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// Reformat from NCHW_VECT_C to NCHW.
//
// Prerequisites: the last dimension of the input must be of size 4.
absl::StatusOr<xla::XlaOp> NCHW_VECT_CToNCHW(xla::XlaOp input);

// Reformat from NCHW to NCHW_VECT_C.
//
// Prerequisites: the vectorized dimension `C` must be a multiple of 4.
absl::StatusOr<xla::XlaOp> NCHWToNCHW_VECT_C(xla::XlaOp input);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_DATA_FORMAT_H_
