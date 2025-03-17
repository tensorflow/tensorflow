/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_HLO_BUILDER_LIB_TRIDIAGONAL_H_
#define XLA_HLO_BUILDER_LIB_TRIDIAGONAL_H_

#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace tridiagonal {

enum SolverAlgorithm { kThomas };

absl::StatusOr<XlaOp> TridiagonalSolver(SolverAlgorithm algo,
                                        XlaOp lower_diagonal,
                                        XlaOp main_diagonal,
                                        XlaOp upper_diagonal, XlaOp rhs);

absl::StatusOr<XlaOp> TridiagonalSolver(SolverAlgorithm algo, XlaOp diagonals,
                                        XlaOp rhs);

absl::StatusOr<XlaOp> TridiagonalMatMul(XlaOp upper_diagonal,
                                        XlaOp main_diagonal,
                                        XlaOp lower_diagonal, XlaOp rhs);

}  // namespace tridiagonal
}  // namespace xla

#endif  // XLA_HLO_BUILDER_LIB_TRIDIAGONAL_H_
