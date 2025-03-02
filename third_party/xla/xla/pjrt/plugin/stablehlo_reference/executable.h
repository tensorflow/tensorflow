/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_EXECUTABLE_H_
#define XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_EXECUTABLE_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/computation_placer.h"

namespace mlir::stablehlo {

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
StablehloReferenceCompile(mlir::ModuleOp module,
                          xla::DeviceAssignment assignment,
                          xla::PjRtClient* client);

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
StablehloReferenceCompile(xla::XlaComputation const& computation,
                          xla::DeviceAssignment assignment,
                          xla::PjRtClient* client);

}  // namespace mlir::stablehlo

#endif  // XLA_PJRT_PLUGIN_STABLEHLO_REFERENCE_EXECUTABLE_H_
