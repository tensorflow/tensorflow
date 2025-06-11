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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_STABLEHLO_UTILS_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_STABLEHLO_UTILS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::spmd {
// Converts StableHLO module (with Shardy dialect) to XLA HLO.
absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertShardyToHlo(
    mlir::ModuleOp shardy_stablehlo_module);

}  // namespace xla::spmd

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_STABLEHLO_UTILS_H_
