/* Copyright 2024 The JAX Authors.

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

#include "xla/mosaic/dialect/tpu/transforms/apply_vector_layout_extensions.h"

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"

namespace mlir::tpu::extensions {

using RewriteContext = ApplyVectorLayoutContext;

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, ArrayRef<Layout>)>;

const llvm::StringMap<rule_type> &rules() {
  static const llvm::StringMap<rule_type> *rules =
      new llvm::StringMap<rule_type>{};
  return *rules;
}

}  // namespace mlir::tpu::extensions