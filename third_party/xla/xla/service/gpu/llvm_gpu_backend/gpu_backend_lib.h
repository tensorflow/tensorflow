/* Copyright 2017 The OpenXLA Authors.

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

// LLVM-based compiler backend.
#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "llvm/PassRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry);

// Returns the TargetMachine, given a triple.
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, absl::string_view cpu_name,
    const DebugOptions& debug_options, absl::string_view feature_str);

// Returns whether the module could use any device bitcode library functions.
bool CouldNeedDeviceBitcode(const llvm::Module& module);

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
absl::Status LinkWithBitcodeVector(
    llvm::Module* module, const std::vector<std::string>& bitcode_path_vector);

using TargetModuleLinker = std::function<absl::Status(
    llvm::Module*, stream_executor::GpuComputeCapability, const DebugOptions&,
    const std::string&)>;

// Links and optimizes the module.
absl::Status LinkAndOptimizeModule(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, const std::string& device_bitcode_path,
    TargetModuleLinker module_linker, llvm::Triple default_target_triple,
    llvm::TargetMachine* target_machine, int inline_threshold);

namespace nvptx {
// Gets the GPU name as it's known to LLVM for a given compute
// capability.  If we see an unrecognized compute capability, we
// return the highest one that is known and below the selected device.
std::string GetSmName(
    stream_executor::CudaComputeCapability compute_capability);

// Compiles the argument module and returns it. libdevice_dir_path is the
// parent directory of the libdevice bitcode libraries. The contents of the
// module may be changed.
//
// The Compile.* interfaces each create their own llvm::LLVMContext objects
// for thread safety, but note that LLVM's multithreaded support is very
// preliminary; multithreaded use is not recommended at this time.
absl::StatusOr<std::string> CompileToPtx(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    std::function<void(llvm::TargetMachine*)> configure_target = nullptr);

// Determine PTX version from CUDA version.
stream_executor::SemanticVersion
DetermineHighestSupportedPtxVersionFromCudaVersion(
    stream_executor::SemanticVersion cuda_version);

}  // namespace nvptx

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_GPU_BACKEND_LIB_H_
