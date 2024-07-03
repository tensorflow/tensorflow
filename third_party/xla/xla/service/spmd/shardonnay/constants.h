/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDONNAY_CONSTANTS_H_
#define XLA_SERVICE_SPMD_SHARDONNAY_CONSTANTS_H_

#include "llvm/ADT/StringRef.h"

namespace xla {
namespace sdy {

// The attribute name for xla::HloSharding.
inline constexpr llvm::StringRef kXlaShardingAttr = "mhlo.sharding";

// The target name of the Sharding custom call.
inline constexpr llvm::StringRef kShardingCustomCallTargetName = "Sharding";

// The target name of the SPMDFullToShardShape custom call.
inline constexpr llvm::StringRef kSPMDFullToShardShapeCallTargetName =
    "SPMDFullToShardShape";

// The target name of the SPMDShardToFullShape custom call.
inline constexpr llvm::StringRef kSPMDShardToFullShapeCallTargetName =
    "SPMDShardToFullShape";

// The attribute name for backend config.
inline constexpr llvm::StringRef kXlaBackendConfigAttr = "backend_config";

// Attribute name for temporarily storing the Shardonnay sharding during HLO
// round-trip. It cannot match the name kShardingAttr ("sdy.sharding"), as
// during round-trip, going from HLO to MHLO, the code removes attributes
// in the `frontend_attributes` field, making them top level. And Shardonnay
// verification expects `kShardingAttr` to be of type
// TensorShardingAttr/TensorShardingPerValueAttr - not a StringAttr.
inline constexpr llvm::StringRef kShardingRoundTripAttr = "xla.sdy.sharding";

// Attribute name for temporarily storing the Shardonnay meshes during HLO
// round-trip.
inline constexpr llvm::StringRef kMeshesRoundTripAttr = "xla.sdy.meshes";

// The target name of the custom call when round tripping during HLO
// round-trip.
inline constexpr llvm::StringRef kFuncResultShardingTargetName =
    "xla.sdy.FuncResultSharding";

// Attribute name for storing frontend attributes in XLA.
inline constexpr llvm::StringRef kFrontendAttributesAttr =
    "mhlo.frontend_attributes";

// Attribute name for determining whether the frontend Python framework has
// lowered to SDY collectives and has exported them using
// `SdyRoundTripExportPipeline`.
// TODO(bartchr): remove this when JAX & PartIR integration is complete.
inline constexpr llvm::StringRef kPythonIntegrationComplete =
    "xla.sdy.python_integration_complete";

// Attribute name for determining whether tuple parameters should be used for
// the rest of the XLA pipeline.
// TODO(b/345414638): remove this when Shardonnay is the first thing run in the
// XLA pipeline, so no HLO<->MLIR round-tripping.
inline constexpr llvm::StringRef kUseTupleArgs = "xla.sdy.use_tuple_args";

// The name of the global mesh.
inline constexpr llvm::StringRef kGlobalMeshName = "mesh";

}  //  namespace sdy
}  //  namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDONNAY_CONSTANTS_H_
