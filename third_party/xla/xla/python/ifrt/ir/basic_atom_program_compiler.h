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

#ifndef XLA_PYTHON_IFRT_IR_BASIC_ATOM_PROGRAM_COMPILER_H_
#define XLA_PYTHON_IFRT_IR_BASIC_ATOM_PROGRAM_COMPILER_H_

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {
namespace ifrt {

// Compiles atom programs for IFRT IR programs. Fulfills the contract required
// by `IfrtIrProgramCompiler`.
class BasicAtomProgramCompiler final : public AtomProgramCompiler {
 public:
  static absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> Create(
      Client* absl_nonnull client,
      absl::Span<const DeviceId> device_assignments);

  tsl::Future<LoadedExecutableRef> CompileXla(
      std::unique_ptr<HloProgram> hlo_program,
      xla::CompileOptions options) final;

  tsl::Future<LoadedExecutableRef> CompileMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) final;

 private:
  BasicAtomProgramCompiler(Client* absl_nonnull client,
                           absl::Span<const DeviceId> device_assignments);

  Client* absl_nonnull const client_;
  const std::vector<DeviceId> device_assignments_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_BASIC_ATOM_PROGRAM_COMPILER_H_
