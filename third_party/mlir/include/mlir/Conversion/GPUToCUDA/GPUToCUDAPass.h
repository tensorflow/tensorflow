//===- GPUToCUDAPass.h - MLIR CUDA runtime support --------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOCUDA_GPUTOCUDAPASS_H_
#define MLIR_CONVERSION_GPUTOCUDA_GPUTOCUDAPASS_H_

#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlir {

class Location;
class ModuleOp;

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

template <typename T> class OpPassBase;

using OwnedCubin = std::unique_ptr<std::vector<char>>;
using CubinGenerator =
    std::function<OwnedCubin(const std::string &, Location, StringRef)>;

/// Creates a pass to convert kernel functions into CUBIN blobs.
///
/// This transformation takes the body of each function that is annotated with
/// the 'nvvm.kernel' attribute, copies it to a new LLVM module, compiles the
/// module with help of the nvptx backend to PTX and then invokes the provided
/// cubinGenerator to produce a binary blob (the cubin). Such blob is then
/// attached as a string attribute named 'nvvm.cubin' to the kernel function.
/// After the transformation, the body of the kernel function is removed (i.e.,
/// it is turned into a declaration).
std::unique_ptr<OpPassBase<ModuleOp>>
createConvertGPUKernelToCubinPass(CubinGenerator cubinGenerator);

/// Creates a pass to convert a gpu.launch_func operation into a sequence of
/// CUDA calls.
///
/// This pass does not generate code to call CUDA directly but instead uses a
/// small wrapper library that exports a stable and conveniently typed ABI
/// on top of CUDA.
std::unique_ptr<OpPassBase<ModuleOp>>
createConvertGpuLaunchFuncToCudaCallsPass();

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOCUDA_GPUTOCUDAPASS_H_
