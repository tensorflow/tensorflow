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

#ifndef XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_PIPELINE_PASSES_H_
#define XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_PIPELINE_PASSES_H_

#include "mlir/Pass/PassOptions.h"

namespace xla {
namespace sdy {

// Adds the common import passes for both the SDY and MHLO import
// pipelines that need to be called before each pass converts an HLO sharding/
// SDY sharding string into an `sdy.sharding` attribute.
void addCommonPreImportPasses(mlir::OpPassManager& pm);

// Adds the common import passes for both the SDY and MHLO import
// pipelines that need to be called after each pass converts an HLO sharding/
// SDY sharding string into an `sdy.sharding` attribute.
void addCommonPostImportPasses(mlir::OpPassManager& pm);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_ROUND_TRIP_COMMON_PIPELINE_PASSES_H_
