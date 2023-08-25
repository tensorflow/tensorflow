/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TOPK_SPLITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TOPK_SPLITTER_H_

#include <cstddef>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

// Splits large TopK into batches of smaller TopKs, followed by sorting and
// slicing the results of those smaller topks. We consider TopKs to be 'large'
// the last dimension of the TopK is larger than `split_threshold`.
class TopKSplitter : public HloModulePass {
 public:
  explicit TopKSplitter(size_t split_threshold = 1024 * 1024)
      : split_threshold_(split_threshold) {}
  absl::string_view name() const override { return "topk-splitter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const size_t split_threshold_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TOPK_SPLITTER_H_
