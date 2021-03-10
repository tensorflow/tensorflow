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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_QR_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_QR_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Computes the QR decompositions of a batch of matrices. That is,
// given a (batched) matrix a, computes an orthonormal matrix Q and an
// upper-triangular matrix R such that a = QR.
// `a` must be a (batched) matrix of size [..., m, n].
struct QrDecomposition {
  // A matrix with the same shape as the input matrix `a`, whose upper triangle
  // (inclusive of the diagonal) is the matrix R, and whose lower triangle
  // (exclusive of the diagonal) contains the elementary Householder reflectors.
  // This is the same output format as used by LAPACK's xGEQRF routine.
  XlaOp q_and_r;
  // A vector of shape [..., min(m, n)] containing the scalar factors of the
  // elementary Householder reflectors.
  XlaOp taus;
};

QrDecomposition Qr(XlaOp a);

// Given `a` and `taus` as returned by `QRDecomposition`, compute the product of
// the elementary Householder reflectors (i.e., the matrix Q of the QR
// decomposition). The equivalent LAPACK routine is xORGQR/xUNGQR.
XlaOp ProductOfElementaryHouseholderReflectors(XlaOp a, XlaOp taus);

// Helper that combines `Qr` and `ProductOfElementaryHouseholderReflectors` to
// compute explicit matrices `q` and `r`.
void QrExplicit(XlaOp a, bool full_matrices, XlaOp& q, XlaOp& r);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_QR_H_
