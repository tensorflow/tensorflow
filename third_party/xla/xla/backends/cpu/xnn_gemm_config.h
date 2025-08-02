/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_XNN_GEMM_CONFIG_H_
#define XLA_BACKENDS_CPU_XNN_GEMM_CONFIG_H_

#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/primitive_util.h"

namespace xla::cpu {

struct XnnGemm {
  DotCanonicalDims dot_canonical_dims;
  PrimitiveType lhs_dtype;
  PrimitiveType rhs_dtype;
  PrimitiveType out_dtype;
};

// XnnGemmConfig is a static lightweight  mechanism for determining if a given
// gemm should be offloaded to XNNPACK vs handled by OneDNN/Eigen.
// Currently it uses a classifier - neural network with: 6 input features
// m, k, n, log(m), log(k), log(n), two hidden layers of size 8 and a cut-off
// threshold for the predicted probability tuned to keep the false positive rate
// below 1%. The classifier was trained on synthetic data (20K random gemms).
// TODO(ashaposhnikov): add a reference to documentation / collab.
class XnnGemmConfig {
  mutable std::function<bool(const XnnGemm&)> test_filter_ = nullptr;

 public:
  XnnGemmConfig() = default;

  enum class Opinion { kAccept, kReject, kNoIdea };

  Opinion Evaluate(const XnnGemm& xnn_gemm,
                   const TargetMachineFeatures* cpu_features) const;

  template <typename Filter>
  void SetTestFilter(Filter&& test_filter) const {
    test_filter_ = std::forward<Filter>(test_filter);
  }
};

const XnnGemmConfig& GetXnnGemmConfig();

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_XNN_GEMM_CONFIG_H_
