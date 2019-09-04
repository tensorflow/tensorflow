//===- GenerateCubinAccessors.cpp - MLIR GPU lowering passes --------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a pass to generate LLVMIR functions that return the
// data stored in nvvm.cubin char* blob.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace {

// TODO(herhut): Move to shared location.
constexpr const char *kCubinAnnotation = "nvvm.cubin";
constexpr const char *kCubinGetterAnnotation = "nvvm.cubingetter";
constexpr const char *kCubinGetterSuffix = "_cubin";
constexpr const char *kCubinStorageSuffix = "_cubin_cst";

/// A pass generating global strings and getter functions for all cubin blobs
/// annotated on functions via the nvvm.cubin attribute.
class GpuGenerateCubinAccessorsPass
    : public ModulePass<GpuGenerateCubinAccessorsPass> {
private:
  LLVM::LLVMType getIndexType() {
    unsigned bits =
        llvmDialect->getLLVMModule().getDataLayout().getPointerSizeInBits();
    return LLVM::LLVMType::getIntNTy(llvmDialect, bits);
  }

  // Inserts a global constant string containing `blob` into the parent module
  // of `orig` and generates the function that returns the address of the first
  // character of this string.
  // TODO(herhut): consider fusing this pass with launch-func-to-cuda.
  void generate(FuncOp orig, StringAttr blob) {
    Location loc = orig.getLoc();
    SmallString<128> nameBuffer(orig.getName());
    auto module = orig.getParentOfType<ModuleOp>();
    assert(module && "function must belong to a module");

    // Insert the getter function just after the original function.
    OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    moduleBuilder.setInsertionPoint(orig.getOperation()->getNextNode());
    auto getterType = moduleBuilder.getFunctionType(
        llvm::None, LLVM::LLVMType::getInt8PtrTy(llvmDialect));
    nameBuffer.append(kCubinGetterSuffix);
    auto result = moduleBuilder.create<FuncOp>(
        loc, StringRef(nameBuffer), getterType, ArrayRef<NamedAttribute>());
    Block *entryBlock = result.addEntryBlock();

    // Drop the getter suffix before appending the storage suffix.
    nameBuffer.resize(orig.getName().size());
    nameBuffer.append(kCubinStorageSuffix);

    // Obtain the address of the first character of the global string containing
    // the cubin and return from the getter.
    OpBuilder builder(entryBlock);
    Value *startPtr = LLVM::createGlobalString(
        loc, builder, StringRef(nameBuffer), blob.getValue(), llvmDialect);
    builder.create<LLVM::ReturnOp>(loc, startPtr);

    // Store the name of the getter on the function for easier lookup.
    orig.setAttr(kCubinGetterAnnotation, builder.getSymbolRefAttr(result));
  }

public:
  // Perform the conversion on the module.  This may insert globals, so it
  // cannot be done on multiple functions in parallel.
  void runOnModule() override {
    llvmDialect =
        getModule().getContext()->getRegisteredDialect<LLVM::LLVMDialect>();

    for (auto func : getModule().getOps<FuncOp>()) {
      StringAttr cubinBlob = func.getAttrOfType<StringAttr>(kCubinAnnotation);
      if (!cubinBlob)
        continue;
      generate(func, cubinBlob);
    }
  }

private:
  LLVM::LLVMDialect *llvmDialect;
};

} // anonymous namespace

std::unique_ptr<ModulePassBase> createGenerateCubinAccessorPass() {
  return std::make_unique<GpuGenerateCubinAccessorsPass>();
}

static PassRegistration<GpuGenerateCubinAccessorsPass>
    pass("generate-cubin-accessors",
         "Generate LLVMIR functions that give access to cubin data");

} // namespace mlir
