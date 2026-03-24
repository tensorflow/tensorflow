/* Copyright 2025 The OpenXLA Authors.

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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/debug_options_flags.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/init_main.h"

namespace xla::gpu::nvptx {

absl::Status Run(const std::string& arch, const std::string& input_ll_path) {
  if (input_ll_path.empty()) {
    return absl::InvalidArgumentError("Input file path is required.");
  }
  if (arch.empty()) {
    return absl::InvalidArgumentError("--arch is required.");
  }

  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> module = LoadIRModule(input_ll_path, &ctx);
  if (!module) {
    return absl::InternalError(
        absl::StrCat("Failed to load module from ", input_ll_path));
  }

  // Create a GpuComputeCapability from the arch flag.
  auto cuda_compute_capability =
      stream_executor::CudaComputeCapability::FromString(arch);
  if (!cuda_compute_capability.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid GPU architecture: ", arch));
  }

  stream_executor::GpuComputeCapability gpu_version(*cuda_compute_capability);

  // Get DebugOptions.
  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  auto llvm_opts = GetNVPTXBackendOptions(debug_options);

  // Compile to PTX.
  auto ptx_or = CompileToPtx(module.get(), gpu_version, debug_options);
  std::cout << *ptx_or << std::endl;
  return absl::OkStatus();
}

}  // namespace xla::gpu::nvptx

int main(int argc, char* argv[]) {
  std::string arch;
  std::vector<tsl::Flag> flag_list = {tsl::Flag(
      "arch", &arch,
      "The GPU architecture to target, e.g., '8.6' or '9.0a' or '10.0f'.")};
  xla::AppendDebugOptionsFlags(&flag_list);

  const std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_result = tsl::Flags::Parse(&argc, argv, flag_list);

  tsl::port::InitMain(usage.c_str(), &argc, &argv);
  if (!parse_result || argc != 2) {
    std::cerr << usage << std::endl;
    return 1;
  }
  absl::Status status = xla::gpu::nvptx::Run(arch, argv[1]);
  if (!status.ok()) {
    std::cerr << "Error: " << status << std::endl;
    return 1;
  }
  return 0;
}
