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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ASYNC_OP_CANONICALIZER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ASYNC_OP_CANONICALIZER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// This pass looks at all of the async operations in the module and assigns the
// async operations that participate in the same async action a unique async
// group id. Async operations in the same group id typically consist of one
// async-start operation, one async-done operation, and zero or more
// async-update operations. Then, this pass ensures all of the async operations
// with the same group id wrap the same computation such that each async
// computation is associated with all of the async operations that have the same
// group id.
class AsyncOpCanonicalizer : public HloModulePass {
 public:
  ~AsyncOpCanonicalizer() override = default;
  absl::string_view name() const override { return "async-op-canonicalizer"; }
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ASYNC_OP_CANONICALIZER_H_
