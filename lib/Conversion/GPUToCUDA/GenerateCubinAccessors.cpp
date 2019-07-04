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

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace {

// TODO(herhut): Move to shared location.
constexpr const char *kCubinAnnotation = "nvvm.cubin";
constexpr const char *kCubinGetterAnnotation = "nvvm.cubingetter";
constexpr const char *kCubinGetterSuffix = "_cubin";
constexpr const char *kMallocHelperName = "malloc";

/// A pass generating getter functions for all cubin blobs annotated on
/// functions via the nvvm.cubin attribute.
///
/// The functions allocate memory using the system malloc call with signature
/// void *malloc(size_t size). This function has to be provided by the actual
/// runner that executes the generated code.
///
/// This is a stop-gap measure until MLIR supports global constants.
class GpuGenerateCubinAccessorsPass
    : public ModulePass<GpuGenerateCubinAccessorsPass> {
private:
  LLVM::LLVMType getIndexType() {
    unsigned bits =
        llvmDialect->getLLVMModule().getDataLayout().getPointerSizeInBits();
    return LLVM::LLVMType::getIntNTy(llvmDialect, bits);
  }

  Function getMallocHelper(Location loc, Builder &builder) {
    Function result = getModule().getNamedFunction(kMallocHelperName);
    if (!result) {
      result = Function::create(
          loc, kMallocHelperName,
          builder.getFunctionType(ArrayRef<Type>{getIndexType()},
                                  LLVM::LLVMType::getInt8PtrTy(llvmDialect)));
      getModule().push_back(result);
    }
    return result;
  }

  // Generates a function that returns a char array at runtime that contains the
  // data from blob. As there are currently no global constants, this uses a
  // sequence of store operations.
  // TODO(herhut): Use global constants instead.
  Function generateCubinAccessor(Builder &builder, Function &orig,
                                 StringAttr blob) {
    Location loc = orig.getLoc();
    SmallString<128> nameBuffer(orig.getName());
    nameBuffer.append(kCubinGetterSuffix);
    // Generate a function that returns void*.
    Function result = Function::create(
        loc, mlir::Identifier::get(nameBuffer, &getContext()),
        builder.getFunctionType(ArrayRef<Type>{},
                                LLVM::LLVMType::getInt8PtrTy(llvmDialect)));
    // Insert a body block that just returns the constant.
    OpBuilder ob(result.getBody());
    ob.createBlock();
    auto sizeConstant = ob.create<LLVM::ConstantOp>(
        loc, getIndexType(),
        builder.getIntegerAttr(builder.getIndexType(), blob.getValue().size()));
    auto memory =
        ob.create<LLVM::CallOp>(
              loc, ArrayRef<Type>{LLVM::LLVMType::getInt8PtrTy(llvmDialect)},
              builder.getFunctionAttr(getMallocHelper(loc, builder)),
              ArrayRef<Value *>{sizeConstant})
            .getResult(0);
    for (auto byte : llvm::enumerate(blob.getValue().bytes())) {
      auto index = ob.create<LLVM::ConstantOp>(
          loc, LLVM::LLVMType::getInt32Ty(llvmDialect),
          builder.getI32IntegerAttr(byte.index()));
      auto gep =
          ob.create<LLVM::GEPOp>(loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect),
                                 memory, ArrayRef<Value *>{index});
      auto value = ob.create<LLVM::ConstantOp>(
          loc, LLVM::LLVMType::getInt8Ty(llvmDialect),
          builder.getIntegerAttr(builder.getIntegerType(8), byte.value()));
      ob.create<LLVM::StoreOp>(loc, value, gep);
    }
    ob.create<LLVM::ReturnOp>(loc, ArrayRef<Value *>{memory});
    // Store the name of the getter on the function for easier lookup.
    orig.setAttr(kCubinGetterAnnotation, builder.getFunctionAttr(result));
    return result;
  }

public:
  // Run the dialect converter on the module.
  void runOnModule() override {
    llvmDialect =
        getModule().getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    auto module = getModule();
    Builder builder(&getContext());

    auto functions = module.getFunctions();
    for (auto it = functions.begin(); it != functions.end();) {
      // Move iterator to after the current function so that potential insertion
      // of the accessor is after the kernel with cubin iself.
      Function orig = *it++;
      StringAttr cubinBlob = orig.getAttrOfType<StringAttr>(kCubinAnnotation);
      if (!cubinBlob)
        continue;
      module.insert(it, generateCubinAccessor(builder, orig, cubinBlob));
    }
  }

private:
  LLVM::LLVMDialect *llvmDialect;
};

} // anonymous namespace

ModulePassBase *createGenerateCubinAccessorPass() {
  return new GpuGenerateCubinAccessorsPass();
}

static PassRegistration<GpuGenerateCubinAccessorsPass>
    pass("generate-cubin-accessors",
         "Generate LLVMIR functions that give access to cubin data");

} // namespace mlir
