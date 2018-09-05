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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_CHOLESKY_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_CHOLESKY_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace tensorflow {

// Computes the Cholesky decompositions of a batch of symmetric positive
// definite matrices.
// `a` must be a (batched) square matrix; i.e., it must have rank >= 2 with the
// two minor dimensions equal.
// The algorithm implements a blocked Cholesky decomposition; `block_size` is
// the block size to use.
// TODO(phawkins): check for negative values on the diagonal and return an
// error, instead of silently yielding NaNs.
// TODO(znado): handle the complex Hermitian case
xla::XlaOp Cholesky(xla::XlaOp a, int64 block_size = 256,
                    xla::PrecisionConfigProto::Precision precision =
                        xla::PrecisionConfigProto::HIGHEST);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_CHOLESKY_H_
