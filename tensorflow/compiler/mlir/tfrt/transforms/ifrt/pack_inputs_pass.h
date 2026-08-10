/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_PACK_INPUTS_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_PACK_INPUTS_PASS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace tensorflow {
namespace ifrt_serving {

struct SliceInfo {
  unsigned arg_index;
  int64_t start;
  int64_t size;
};

// This pass packs specified tensor inputs of main function into a single i8
// tensor buffer using explicit slice coordinates.
class PackInputsPass
    : public mlir::PassWrapper<PackInputsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackInputsPass)

  PackInputsPass() = default;
  PackInputsPass(const PackInputsPass& other)
      : PassWrapper(other), slices_(other.slices_) {}
  explicit PackInputsPass(llvm::ArrayRef<SliceInfo> slices);

  void runOnOperation() final;

  void getDependentDialects(mlir::DialectRegistry& registry) const override;

  mlir::StringRef getArgument() const override;
  mlir::StringRef getDescription() const override;

 private:
  std::vector<SliceInfo> slices_;
  ListOption<int64_t> slices_flat_list_{
      *this, "slices",
      llvm::cl::desc(
          "Flat list of integers specifying {arg_index, start, size} "
          "for each slice to be packed.")};
};

// Creates an instance of the PackInputsPass.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreatePackInputsPass(
    llvm::ArrayRef<SliceInfo> slices = {});

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_PACK_INPUTS_PASS_H_
