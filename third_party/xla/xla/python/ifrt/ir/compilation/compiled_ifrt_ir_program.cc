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

#include "xla/python/ifrt/ir/compilation/compiled_ifrt_ir_program.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compilation/program_lowering.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/program_interpreter.h"
#include "xla/python/ifrt/with_user_context.h"
#include "xla/tsl/concurrency/future.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {

using ::tsl::profiler::TraceMe;

tsl::Future<std::shared_ptr<CompiledIfrtIrProgram>>
CompiledIfrtIrProgram::Create(
    std::unique_ptr<IfrtIRProgram> ifrt_ir_program,
    std::unique_ptr<IfrtIRCompileOptions> ifrt_ir_compile_options,
    Client* client,
    std::shared_ptr<AtomProgramCompiler> atom_program_compiler) {
  TraceMe traceme("ProgramCompiler::CompileForInterpreter");

  // `client` outlives all compilation tasks per `xla::ifrt::Compiler`'s
  // lifetime contract (callers must await all returned compilation futures
  // before destroying the compiler and client).
  return LowerIfrtIrProgram(std::move(ifrt_ir_program),
                            std::move(ifrt_ir_compile_options), client,
                            std::move(atom_program_compiler),
                            AtomProgramLoadingMode::kEager)
      .Map(WithCurrentUserContext(
          [client](LoweredIfrtIrProgramState state)
              -> tsl::Future<std::shared_ptr<CompiledIfrtIrProgram>> {
            std::shared_ptr<AtomExecutableFutureMap>
                atom_executable_future_map =
                    std::move(state.atom_executable_future_map);
            CHECK_NE(atom_executable_future_map, nullptr);

            // Perform the rest of the work once all atom programs are
            // successfully compiled since they need information from the
            // compiled executables.
            std::vector<tsl::Future<>> ready_futures;
            ready_futures.reserve(atom_executable_future_map->size());
            // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
            for (const auto& [key, exec] : *atom_executable_future_map) {
              ready_futures.push_back(exec.GetReadyFuture());
            }
            auto create_program = [state = std::move(state),
                                   atom_executable_future_map =
                                       std::move(atom_executable_future_map),
                                   client]() mutable
                -> absl::StatusOr<std::shared_ptr<CompiledIfrtIrProgram>> {
              auto atom_executable_map = std::make_shared<AtomExecutableMap>();
              // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
              for (const auto& [key, exec] : *atom_executable_future_map) {
                CHECK(exec.IsReady());
                ABSL_ASSIGN_OR_RETURN(LoadedExecutableRef executable, exec.Await());
                if (executable == nullptr) {
                  return absl::InternalError(
                      absl::StrCat("Compiled atom program `", key,
                                   "` returned a null executable"));
                }
                atom_executable_map->insert({key, std::move(executable)});
              }

              absl::Status layout_status = PopulateIfrtIrProgramLayouts(
                  state.ifrt_ir_program->mlir_module, client,
                  *atom_executable_map, /*compiled_atom_programs=*/nullptr,
                  absl::MakeSpan(state.in_specs),
                  absl::MakeSpan(state.out_specs), state.device_list);

              ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ProgramInterpreter> interpreter,
                               ProgramInterpreter::Create(
                                   client, state.program_name,
                                   state.ifrt_ir_program->mlir_module,
                                   atom_executable_map, state.device_list));
              ABSL_ASSIGN_OR_RETURN(ProgramInterpreter::ExecuteFn execute_fn,
                               interpreter->BuildExecuteFn());

              return std::make_shared<CompiledIfrtIrProgram>(
                  CompiledIfrtIrProgram{
                      /*program_name=*/std::move(state.program_name),
                      /*atom_program_executables=*/
                      std::move(atom_executable_map),
                      /*in_specs=*/std::move(state.in_specs),
                      /*out_specs=*/std::move(state.out_specs),
                      /*layout_status=*/layout_status,
                      /*donatable_input_indices=*/
                      std::move(state.donatable_input_indices),
                      /*program=*/std::move(state.ifrt_ir_program),
                      /*devices=*/std::move(state.device_list),
                      /*compile_options=*/std::move(state.compile_options),
                      /*execute_fn=*/std::move(execute_fn),
                  });
            };
            return tsl::JoinFutures(ready_futures)
                .Map(GetIfrtIrCompilationExecutor(),
                     WithCurrentUserContext(std::move(create_program)));
          }));
}

}  // namespace ifrt
}  // namespace xla
