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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_NUMERIC_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_NUMERIC_H_

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Returns a rank 1 tensor of `type` containing values [0, 1, 2, ...].
XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64 size);

// Returns an m x n matrix with 1s on the diagonal elements, zeros everywhere
// else.
XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64 m, int64 n);

// Get the diagonals of the last two dimensions. If 'x' has shape
// [..., M, N], then the output has shape [..., min(M, N)], containing the
// diagonal elements (i.e., with indices [..., i, i]).
XlaOp GetMatrixDiagonal(XlaOp x);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_NUMERIC_H_
