/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_STOCHASTIC_CONVERT_DECOMPOSER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_STOCHASTIC_CONVERT_DECOMPOSER_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// StochasticConvertDecomposer is a pass which replaces unsupported
// stochastic-convert with multiple hlos.
class StochasticConvertDecomposer : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "stochastic_convert_decomposer";
  }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_STOCHASTIC_CONVERT_DECOMPOSER_H_
