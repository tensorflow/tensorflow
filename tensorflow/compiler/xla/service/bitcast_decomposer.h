/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BITCAST_DECOMPOSER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BITCAST_DECOMPOSER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Decomposes expensive bitcasts inside of fusions into cheap-bitcast+transpose.
//
// Bitcasts outside of fusions are free in XLA:GPU, but bitcasts inside of
// fusions are expensive if !ReshapeIsBitcast.
//
// For each bitcast inside of a fusion where the bitcast has !ReshapeIsBitcast,
// this pass decomposes the bitcast into a sequence of reshape-is-bitcast plus
// transpose-is-bitcast operations.
//
// (In theory, we could do this decomposition logic inside of codegen itself,
// without requiring any modifications to the IR. But it's easier to express it
// in the IR as a late "codegen prepare" pass.)
class BitcastDecomposer : public HloModulePass {
 public:
  absl::string_view name() const override { return "bitcast_decomposer"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BITCAST_DECOMPOSER_H_
