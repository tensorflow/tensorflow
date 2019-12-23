//===- ConvertLaunchFuncToCudaCalls.cpp - MLIR CUDA lowering passes -------===//
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
// This file implements a pass to convert gpu.launch_func op into a sequence of
// CUDA runtime calls. As the CUDA runtime does not have a stable published ABI,
// this pass uses a slim runtime layer that builds on top of the public API from
// the CUDA headers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

// To avoid name mangling, these are defined in the mini-runtime file.
static constexpr const char *cuModuleLoadName = "mcuModuleLoad";
static constexpr const char *cuModuleGetFunctionName = "mcuModuleGetFunction";
static constexpr const char *cuLaunchKernelName = "mcuLaunchKernel";
static constexpr const char *cuGetStreamHelperName = "mcuGetStreamHelper";
static constexpr const char *cuStreamSynchronizeName = "mcuStreamSynchronize";
static constexpr const char *kMcuMemHostRegister = "mcuMemHostRegister";

static constexpr const char *kCubinAnnotation = "nvvm.cubin";
static constexpr const char *kCubinStorageSuffix = "_cubin_cst";

namespace {

/// A pass to convert gpu.launch_func operations into a sequence of CUDA
/// runtime calls.
///
/// In essence, a gpu.launch_func operations gets compiled into the following
/// sequence of runtime calls:
///
/// * mcuModuleLoad        -- loads the module given the cubin data
/// * mcuModuleGetFunction -- gets a handle to the actual kernel function
/// * mcuGetStreamHelper   -- initializes a new CUDA stream
/// * mcuLaunchKernelName  -- launches the kernel on a stream
/// * mcuStreamSynchronize -- waits for operations on the stream to finish
///
/// Intermediate data structures are allocated on the stack.
class GpuLaunchFuncToCudaCallsPass
    : public ModulePass<GpuLaunchFuncToCudaCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    const llvm::Module &module = llvmDialect->getLLVMModule();
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmPointerPointerType = llvmPointerType.getPointerTo();
    llvmInt8Type = LLVM::LLVMType::getInt8Ty(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
    llvmIntPtrType = LLVM::LLVMType::getIntNTy(
        llvmDialect, module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getVoidType() { return llvmVoidType; }

  LLVM::LLVMType getPointerType() { return llvmPointerType; }

  LLVM::LLVMType getPointerPointerType() { return llvmPointerPointerType; }

  LLVM::LLVMType getInt8Type() { return llvmInt8Type; }

  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }

  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  LLVM::LLVMType getIntPtrType() {
    const llvm::Module &module = getLLVMDialect()->getLLVMModule();
    return LLVM::LLVMType::getIntNTy(
        getLLVMDialect(), module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getCUResultType() {
    // This is declared as an enum in CUDA but helpers use i32.
    return getInt32Type();
  }

  // Allocate a void pointer on the stack.
  ValuePtr allocatePointer(OpBuilder &builder, Location loc) {
    auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                                builder.getI32IntegerAttr(1));
    return builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(), one,
                                          /*alignment=*/0);
  }

  void declareCudaFunctions(Location loc);
  ValuePtr setupParamsArray(gpu::LaunchFuncOp launchOp, OpBuilder &builder);
  ValuePtr generateKernelNameConstant(StringRef name, Location loc,
                                      OpBuilder &builder);
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);

public:
  // Run the dialect converter on the module.
  void runOnModule() override {
    // Cache the LLVMDialect for the current module.
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    // Cache the used LLVM types.
    initializeCachedTypes();

    getModule().walk([this](mlir::gpu::LaunchFuncOp op) {
      translateGpuLaunchCalls(op);
    });

    // GPU kernel modules are no longer necessary since we have a global
    // constant with the CUBIN data.
    for (auto m : llvm::make_early_inc_range(getModule().getOps<ModuleOp>()))
      if (m.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelModuleAttrName()))
        m.erase();
  }

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmPointerPointerType;
  LLVM::LLVMType llvmInt8Type;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  LLVM::LLVMType llvmIntPtrType;
};

} // anonymous namespace

// Adds declarations for the needed helper functions from the CUDA wrapper.
// The types in comments give the actual types expected/returned but the API
// uses void pointers. This is fine as they have the same linkage in C.
void GpuLaunchFuncToCudaCallsPass::declareCudaFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());
  if (!module.lookupSymbol(cuModuleLoadName)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, cuModuleLoadName,
        LLVM::LLVMType::getFunctionTy(
            getCUResultType(),
            {
                getPointerPointerType(), /* CUmodule *module */
                getPointerType()         /* void *cubin */
            },
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(cuModuleGetFunctionName)) {
    // The helper uses void* instead of CUDA's opaque CUmodule and
    // CUfunction.
    builder.create<LLVM::LLVMFuncOp>(
        loc, cuModuleGetFunctionName,
        LLVM::LLVMType::getFunctionTy(
            getCUResultType(),
            {
                getPointerPointerType(), /* void **function */
                getPointerType(),        /* void *module */
                getPointerType()         /* char *name */
            },
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(cuLaunchKernelName)) {
    // Other than the CUDA api, the wrappers use uintptr_t to match the
    // LLVM type if MLIR's index type, which the GPU dialect uses.
    // Furthermore, they use void* instead of CUDA's opaque CUfunction and
    // CUstream.
    builder.create<LLVM::LLVMFuncOp>(
        loc, cuLaunchKernelName,
        LLVM::LLVMType::getFunctionTy(
            getCUResultType(),
            {
                getPointerType(),        /* void* f */
                getIntPtrType(),         /* intptr_t gridXDim */
                getIntPtrType(),         /* intptr_t gridyDim */
                getIntPtrType(),         /* intptr_t gridZDim */
                getIntPtrType(),         /* intptr_t blockXDim */
                getIntPtrType(),         /* intptr_t blockYDim */
                getIntPtrType(),         /* intptr_t blockZDim */
                getInt32Type(),          /* unsigned int sharedMemBytes */
                getPointerType(),        /* void *hstream */
                getPointerPointerType(), /* void **kernelParams */
                getPointerPointerType()  /* void **extra */
            },
            /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(cuGetStreamHelperName)) {
    // Helper function to get the current CUDA stream. Uses void* instead of
    // CUDAs opaque CUstream.
    builder.create<LLVM::LLVMFuncOp>(
        loc, cuGetStreamHelperName,
        LLVM::LLVMType::getFunctionTy(getPointerType(), /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(cuStreamSynchronizeName)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, cuStreamSynchronizeName,
        LLVM::LLVMType::getFunctionTy(getCUResultType(),
                                      getPointerType() /* CUstream stream */,
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kMcuMemHostRegister)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kMcuMemHostRegister,
        LLVM::LLVMType::getFunctionTy(getVoidType(),
                                      {
                                          getPointerType(), /* void *ptr */
                                          getInt64Type()    /* int64 sizeBytes*/
                                      },
                                      /*isVarArg=*/false));
  }
}

// Generates a parameters array to be used with a CUDA kernel launch call. The
// arguments are extracted from the launchOp.
// The generated code is essentially as follows:
//
// %array = alloca(numparams * sizeof(void *))
// for (i : [0, NumKernelOperands))
//   %array[i] = cast<void*>(KernelOperand[i])
// return %array
ValuePtr
GpuLaunchFuncToCudaCallsPass::setupParamsArray(gpu::LaunchFuncOp launchOp,
                                               OpBuilder &builder) {
  auto numKernelOperands = launchOp.getNumKernelOperands();
  Location loc = launchOp.getLoc();
  auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                              builder.getI32IntegerAttr(1));
  // Provision twice as much for the `array` to allow up to one level of
  // indirection for each argument.
  auto arraySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(numKernelOperands));
  auto array = builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(),
                                              arraySize, /*alignment=*/0);
  for (unsigned idx = 0; idx < numKernelOperands; ++idx) {
    auto operand = launchOp.getKernelOperand(idx);
    auto llvmType = operand->getType().cast<LLVM::LLVMType>();
    ValuePtr memLocation = builder.create<LLVM::AllocaOp>(
        loc, llvmType.getPointerTo(), one, /*alignment=*/1);
    builder.create<LLVM::StoreOp>(loc, operand, memLocation);
    auto casted =
        builder.create<LLVM::BitcastOp>(loc, getPointerType(), memLocation);

    // Assume all struct arguments come from MemRef. If this assumption does not
    // hold anymore then we `launchOp` to lower from MemRefType and not after
    // LLVMConversion has taken place and the MemRef information is lost.
    // Extra level of indirection in the `array`:
    //   the descriptor pointer is registered via @mcuMemHostRegisterPtr
    if (llvmType.isStructTy()) {
      auto registerFunc =
          getModule().lookupSymbol<LLVM::LLVMFuncOp>(kMcuMemHostRegister);
      auto nullPtr = builder.create<LLVM::NullOp>(loc, llvmType.getPointerTo());
      auto gep = builder.create<LLVM::GEPOp>(loc, llvmType.getPointerTo(),
                                             ArrayRef<ValuePtr>{nullPtr, one});
      auto size = builder.create<LLVM::PtrToIntOp>(loc, getInt64Type(), gep);
      builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{},
                                   builder.getSymbolRefAttr(registerFunc),
                                   ArrayRef<ValuePtr>{casted, size});
      ValuePtr memLocation = builder.create<LLVM::AllocaOp>(
          loc, getPointerPointerType(), one, /*alignment=*/1);
      builder.create<LLVM::StoreOp>(loc, casted, memLocation);
      casted =
          builder.create<LLVM::BitcastOp>(loc, getPointerType(), memLocation);
    }

    auto index = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(idx));
    auto gep = builder.create<LLVM::GEPOp>(loc, getPointerPointerType(), array,
                                           ArrayRef<ValuePtr>{index});
    builder.create<LLVM::StoreOp>(loc, casted, gep);
  }
  return array;
}

// Generates an LLVM IR dialect global that contains the name of the given
// kernel function as a C string, and returns a pointer to its beginning.
// The code is essentially:
//
// llvm.global constant @kernel_name("function_name\00")
// func(...) {
//   %0 = llvm.addressof @kernel_name
//   %1 = llvm.constant (0 : index)
//   %2 = llvm.getelementptr %0[%1, %1] : !llvm<"i8*">
// }
ValuePtr GpuLaunchFuncToCudaCallsPass::generateKernelNameConstant(
    StringRef name, Location loc, OpBuilder &builder) {
  // Make sure the trailing zero is included in the constant.
  std::vector<char> kernelName(name.begin(), name.end());
  kernelName.push_back('\0');

  std::string globalName = llvm::formatv("{0}_kernel_name", name);
  return LLVM::createGlobalString(
      loc, builder, globalName, StringRef(kernelName.data(), kernelName.size()),
      LLVM::Linkage::Internal, llvmDialect);
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the 'nvvm.cubin' attribute of the
// kernel function in the IR.
// While MLIR has no global constants, also expects a cubin getter function in
// an 'nvvm.cubingetter' attribute. Such function is expected to return a
// pointer to the cubin blob when invoked.
// With these given, the generated code in essence is
//
// %0 = call %cubingetter
// %1 = alloca sizeof(void*)
// call %mcuModuleLoad(%2, %1)
// %2 = alloca sizeof(void*)
// %3 = load %1
// %4 = <see generateKernelNameConstant>
// call %mcuModuleGetFunction(%2, %3, %4)
// %5 = call %mcuGetStreamHelper()
// %6 = load %2
// %7 = <see setupParamsArray>
// call %mcuLaunchKernel(%6, <launchOp operands 0..5>, 0, %5, %7, nullptr)
// call %mcuStreamSynchronize(%5)
void GpuLaunchFuncToCudaCallsPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();
  declareCudaFunctions(loc);

  auto zero = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                               builder.getI32IntegerAttr(0));
  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernelModule =
      getModule().lookupSymbol<ModuleOp>(launchOp.getKernelModuleName());
  assert(kernelModule && "expected a kernel module");

  auto cubinAttr = kernelModule.getAttrOfType<StringAttr>(kCubinAnnotation);
  if (!cubinAttr) {
    kernelModule.emitOpError()
        << "missing " << kCubinAnnotation << " attribute";
    return signalPassFailure();
  }

  assert(kernelModule.getName() && "expected a named module");
  SmallString<128> nameBuffer(*kernelModule.getName());
  nameBuffer.append(kCubinStorageSuffix);
  ValuePtr data = LLVM::createGlobalString(
      loc, builder, nameBuffer.str(), cubinAttr.getValue(),
      LLVM::Linkage::Internal, getLLVMDialect());

  // Emit the load module call to load the module data. Error checking is done
  // in the called helper function.
  auto cuModule = allocatePointer(builder, loc);
  auto cuModuleLoad =
      getModule().lookupSymbol<LLVM::LLVMFuncOp>(cuModuleLoadName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getCUResultType()},
                               builder.getSymbolRefAttr(cuModuleLoad),
                               ArrayRef<ValuePtr>{cuModule, data});
  // Get the function from the module. The name corresponds to the name of
  // the kernel function.
  auto cuOwningModuleRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), cuModule);
  auto kernelName = generateKernelNameConstant(launchOp.kernel(), loc, builder);
  auto cuFunction = allocatePointer(builder, loc);
  auto cuModuleGetFunction =
      getModule().lookupSymbol<LLVM::LLVMFuncOp>(cuModuleGetFunctionName);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getCUResultType()},
      builder.getSymbolRefAttr(cuModuleGetFunction),
      ArrayRef<ValuePtr>{cuFunction, cuOwningModuleRef, kernelName});
  // Grab the global stream needed for execution.
  auto cuGetStreamHelper =
      getModule().lookupSymbol<LLVM::LLVMFuncOp>(cuGetStreamHelperName);
  auto cuStream = builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getPointerType()},
      builder.getSymbolRefAttr(cuGetStreamHelper), ArrayRef<ValuePtr>{});
  // Invoke the function with required arguments.
  auto cuLaunchKernel =
      getModule().lookupSymbol<LLVM::LLVMFuncOp>(cuLaunchKernelName);
  auto cuFunctionRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), cuFunction);
  auto paramsArray = setupParamsArray(launchOp, builder);
  auto nullpointer =
      builder.create<LLVM::IntToPtrOp>(loc, getPointerPointerType(), zero);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getCUResultType()},
      builder.getSymbolRefAttr(cuLaunchKernel),
      ArrayRef<ValuePtr>{cuFunctionRef, launchOp.getOperand(0),
                         launchOp.getOperand(1), launchOp.getOperand(2),
                         launchOp.getOperand(3), launchOp.getOperand(4),
                         launchOp.getOperand(5), zero, /* sharedMemBytes */
                         cuStream.getResult(0),        /* stream */
                         paramsArray,                  /* kernel params */
                         nullpointer /* extra */});
  // Sync on the stream to make it synchronous.
  auto cuStreamSync =
      getModule().lookupSymbol<LLVM::LLVMFuncOp>(cuStreamSynchronizeName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getCUResultType()},
                               builder.getSymbolRefAttr(cuStreamSync),
                               ArrayRef<ValuePtr>(cuStream.getResult(0)));
  launchOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToCudaCallsPass() {
  return std::make_unique<GpuLaunchFuncToCudaCallsPass>();
}

static PassRegistration<GpuLaunchFuncToCudaCallsPass>
    pass("launch-func-to-cuda",
         "Convert all launch_func ops to CUDA runtime calls");
