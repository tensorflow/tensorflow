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
#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DECL
#include "xla/codegen/emitters/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass();
std::unique_ptr<mlir::Pass> CreateFlattenTensorsPass();
std::unique_ptr<mlir::Pass> CreateLowerTensorsPass(
    const std::string& target_type = "gpu",
    const std::string& gpu_device_info = "");
std::unique_ptr<mlir::Pass> CreateLowerTensorsPass(
    const stream_executor::DeviceDescription& device_description);
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass(
    const std::string& target_type = "gpu",
    const std::string& gpu_device_info = "");
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass(
    const stream_executor::DeviceDescription& device_description);

#define GEN_PASS_REGISTRATION
#include "xla/codegen/emitters/transforms/passes.h.inc"

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
