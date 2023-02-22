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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MOVE_COPY_TO_USERS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MOVE_COPY_TO_USERS_H_

#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Sink kCopy operations as far down the graph as possible.
class MoveCopyToUsers : public HloModulePass {
 public:
  absl::string_view name() const override { return "move_copy_to_users"; }
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // end namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MOVE_COPY_TO_USERS_H_
