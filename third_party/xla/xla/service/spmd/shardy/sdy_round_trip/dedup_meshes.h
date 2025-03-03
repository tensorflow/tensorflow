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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_DEDUP_MESHES_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_DEDUP_MESHES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Creates the pass that deduplicates any meshes with the same axis sizes (in
// the same order) but different names into a single mesh. The mesh that appears
// first in the module is used as the main mesh for that set of meshes with the
// same axis sizes.
//
// This is needed for JAX export where a module may be saved with a mesh with
// one set of axis names, and then loaded with a different set of axis names.
// Since Shardy can't propagate between meshes, this will make sure propagation
// can happen.
std::unique_ptr<mlir::Pass> createSdyRoundTripDedupMeshesPass();

// Registers the xla-sdy-round-trip-dedup-meshes pass.
void registerSdyRoundTripDedupMeshesPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_DEDUP_MESHES_H_
