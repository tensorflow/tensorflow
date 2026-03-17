/* Copyright 2026 The OpenXLA Authors.
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

#include "xla/service/spmd/shardy/stablehlo_round_trip/unflatten_call_graph.h"

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace xla {
namespace sdy {
namespace {

using ::mlir::ModuleOp;
using ::mlir::StringRef;
using ::mlir::sdy::SdyDialect;

class UnflattenCallGraphPass
    : public mlir::PassWrapper<UnflattenCallGraphPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnflattenCallGraphPass)

  explicit UnflattenCallGraphPass(bool dedupFunctionsFully) {
    this->dedupFunctionsFully = dedupFunctionsFully;
  }

  UnflattenCallGraphPass() = default;

  explicit UnflattenCallGraphPass(const UnflattenCallGraphPass& other) {
    this->dedupFunctionsFully = other.dedupFunctionsFully;
  }

  void runOnOperation() final {}

  StringRef getArgument() const override {
    return "xla-sdy-unflatten-call-graph";
  }

  StringRef getDescription() const override {
    return "Unflattens the call graph.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }

  Option<bool> dedupFunctionsFully{
      *this, "dedup-functions-fully",
      llvm::cl::desc(
          "If true, regardless of the input and output shardings of functions, "
          "it keeps one callee function for each caller function. The default "
          "is false, meaning it will deduplicate only if the input and output "
          "shardings are the same."),
      llvm::cl::init(false)};
};
}  // namespace

std::unique_ptr<mlir::Pass> createUnflattenCallGraphPass(
    bool dedupFunctionsFully) {
  return std::make_unique<UnflattenCallGraphPass>(dedupFunctionsFully);
}

void registerUnflattenCallGraphPass() {
  mlir::registerPass([]() {
    return createUnflattenCallGraphPass(/*dedupFunctionsFully=*/false);
  });
}
}  // namespace sdy
}  // namespace xla
