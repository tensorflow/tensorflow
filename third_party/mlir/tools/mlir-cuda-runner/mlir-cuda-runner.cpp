//===- mlir-cpu-runner.cpp - MLIR CPU Execution Driver---------------------===//
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
// This is a command line utility that executes an MLIR file on the GPU by
// translating MLIR to NVVM/LVVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/JitRunner.h"
#include "mlir/Transforms/DialectConversion.h"

#include "cuda.h"

using namespace mlir;

inline void emit_cuda_error(const llvm::Twine &message, const char *buffer,
                            CUresult error, FuncOp &function) {
  function.emitError(message.concat(" failed with error code ")
                         .concat(llvm::Twine{error})
                         .concat("[")
                         .concat(buffer)
                         .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr, msg)                                        \
  {                                                                            \
    auto _cuda_error = (expr);                                                 \
    if (_cuda_error != CUDA_SUCCESS) {                                         \
      emit_cuda_error(msg, jitErrorBuffer, _cuda_error, function);             \
      return {};                                                               \
    }                                                                          \
  }

OwnedCubin compilePtxToCubin(const std::string ptx, FuncOp &function) {
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0), "cuInit");

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState),
                       "cuLinkCreate");

  RETURN_ON_CUDA_ERROR(
      cuLinkAddData(linkState, CUjitInputType::CU_JIT_INPUT_PTX,
                    const_cast<void *>(static_cast<const void *>(ptx.c_str())),
                    ptx.length(), function.getName().data(), /* kernel name */
                    0,       /* number of jit options */
                    nullptr, /* jit options */
                    nullptr  /* jit option values */
                    ),
      "cuLinkAddData");

  void *cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize),
                       "cuLinkComplete");

  char *cubinAsChar = static_cast<char *>(cubinData);
  OwnedCubin result =
      std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState), "cuLinkDestroy");

  return result;
}

namespace {
// A pass that lowers all Standard and Gpu operations to LLVM dialect. It does
// not lower the GPULaunch operation to actual code but dows translate the
// signature of its kernel argument.
class LowerStandardAndGpuToLLVMAndNVVM
    : public ModulePass<LowerStandardAndGpuToLLVMAndNVVM> {
public:
  void runOnModule() override {
    ModuleOp m = getModule();

    OwningRewritePatternList patterns;
    LLVMTypeConverter converter(m.getContext());
    populateStdToLLVMConversionPatterns(converter, patterns);
    populateGpuToNVVMConversionPatterns(converter, patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    if (failed(applyFullConversion(m, target, patterns, &converter)))
      signalPassFailure();
  }
};
} // end anonymous namespace

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());

  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(static_cast<std::unique_ptr<ModulePassBase>>(
      std::make_unique<LowerStandardAndGpuToLLVMAndNVVM>()));
  pm.addPass(createConvertGPUKernelToCubinPass(&compilePtxToCubin));
  pm.addPass(createGenerateCubinAccessorPass());
  pm.addPass(createConvertGpuLaunchFuncToCudaCallsPass());

  if (failed(pm.run(m)))
    return failure();

  return success();
}

int main(int argc, char **argv) {
  return mlir::JitRunnerMain(argc, argv, &runMLIRPasses);
}
