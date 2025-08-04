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

#ifndef XLA_PJRT_PJRT_COMPILER_HELPER_H_
#define XLA_PJRT_PJRT_COMPILER_HELPER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/service/hlo.pb.h"

namespace xla {

// Wrapper around PjRtCompiler::Compile that prepares the required inputs and
// calls PjRtCompiler::Compile for a StableHLO module.
absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompileMlirModule(
    mlir::ModuleOp module, const CompileOptionsProto& options,
    const PjRtTopologyDescriptionProto& topology);

// Wrapper around PjRtCompiler::Compile that prepares the required inputs and
// calls PjRtCompiler::Compile for an HLO module.
absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompileHloModule(
    const HloModuleProto& hlo_module_proto, const CompileOptionsProto& options,
    const PjRtTopologyDescriptionProto& topology);

}  // namespace xla
#endif  // XLA_PJRT_PJRT_COMPILER_HELPER_H_
