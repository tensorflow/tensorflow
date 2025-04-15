/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_HLO_BUILDER_LIB_LOGDET_H_
#define XLA_HLO_BUILDER_LIB_LOGDET_H_

#include "xla/hlo/builder/xla_builder.h"

namespace xla {

// Computes the sign and logarithm of the absolute value of the determinant
// of a batch of square matrices with shape [..., n, n].
struct SignAndLogDet {
  XlaOp sign;    // Either 1, 0, or -1, depending on the determinant's sign.
  XlaOp logdet;  // log(abs(det(a)).
};
SignAndLogDet SLogDet(XlaOp a);

// For a batch of matrices with shape [..., n, n], return log(det(a)).
// Returns NaN if a matrix has a negative determinant.
XlaOp LogDet(XlaOp a);

}  // namespace xla

#endif  // XLA_HLO_BUILDER_LIB_LOGDET_H_
