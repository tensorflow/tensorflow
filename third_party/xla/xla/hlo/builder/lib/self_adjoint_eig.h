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

#ifndef XLA_HLO_BUILDER_LIB_SELF_ADJOINT_EIG_H_
#define XLA_HLO_BUILDER_LIB_SELF_ADJOINT_EIG_H_

#include <cstdint>

#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"

namespace xla {

// The eigenvalue decomposition of a symmetric matrix, the original matrix is
// recovered by v * w * v_t.
struct SelfAdjointEigResult {
  // The i-th column is the normalized eigenvector corresponding to the
  // eigenvalue w[i]. Will return a matrix object if a is a matrix object.
  XlaOp v;
  // The eigenvalues in ascending order, each repeated according to its
  // multiplicity.
  XlaOp w;
};

SelfAdjointEigResult SelfAdjointEig(XlaOp a, bool lower = true,
                                    int64_t max_iter = 15, float tol = 1e-5,
                                    bool sort_eigenvalues = true);

}  // namespace xla

#endif  // XLA_HLO_BUILDER_LIB_SELF_ADJOINT_EIG_H_
