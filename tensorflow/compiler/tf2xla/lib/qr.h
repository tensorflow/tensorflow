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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_QR_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_QR_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace tensorflow {

// Computes the QR decompositions of a batch of matrices. That is,
// given a (batched) matrix a, computes an orthonormal matrix Q and an
// upper-triangular matrix R such that a = QR.
// `a` must be a (batched) matrix of size [..., m, n].
// The algorithm implements a blocked QR decomposition; `block_size` is
// the block size to use.
// TODO(phawkins): handle the complex case.
struct QRDecompositionResult {
  xla::XlaOp q;
  xla::XlaOp r;
};

xla::StatusOr<QRDecompositionResult> QRDecomposition(
    xla::XlaOp a, bool full_matrices, int64 block_size = 128,
    xla::PrecisionConfigProto::Precision precision =
        xla::PrecisionConfigProto::HIGHEST);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_QR_H_
