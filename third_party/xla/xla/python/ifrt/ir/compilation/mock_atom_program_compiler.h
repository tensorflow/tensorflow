/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_COMPILATION_MOCK_ATOM_PROGRAM_COMPILER_H_
#define XLA_PYTHON_IFRT_IR_COMPILATION_MOCK_ATOM_PROGRAM_COMPILER_H_

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include "absl/base/nullability.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// GMock implementation of `AtomProgramCompiler` for unit tests.
class MockAtomProgramCompiler : public AtomProgramCompiler {
 public:
  MOCK_METHOD(
      tsl::Future<LoadedExecutableRef>, CompileAndLoadXla,
      (std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
       std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks),
      (override));
  MOCK_METHOD(
      tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>,
      CompileXla,
      (std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
       std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks),
      (override));
  MOCK_METHOD(tsl::Future<LoadedExecutableRef>, CompileAndLoadMpmdReshard,
              (std::vector<DType> dtypes, std::vector<Shape> shapes,
               std::vector<IfrtArrayType> in_array_types,
               std::vector<IfrtArrayType> out_array_types),
              (override));
  MOCK_METHOD(
      tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>,
      CompileMpmdReshard,
      (std::vector<DType> dtypes, std::vector<Shape> shapes,
       std::vector<IfrtArrayType> in_array_types,
       std::vector<IfrtArrayType> out_array_types),
      (override));
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILATION_MOCK_ATOM_PROGRAM_COMPILER_H_
