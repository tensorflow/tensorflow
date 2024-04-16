/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_CLIENT_COMPILE_ONLY_CLIENT_H_
#define XLA_CLIENT_COMPILE_ONLY_CLIENT_H_

#include <memory>
#include <vector>

#include "xla/client/client.h"
#include "xla/client/xla_computation.h"
#include "xla/service/compile_only_service.h"
#include "xla/service/compiler.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// An XLA Client specialization for doing ahead-of-time compilation.  This does
// not require (or attempt to instantiate) an execution-capable backend for the
// relevant platform.
class CompileOnlyClient : public Client {
 public:
  explicit CompileOnlyClient(CompileOnlyService* service)
      : Client(service), compiler_service_(service) {}

  CompileOnlyClient(const CompileOnlyClient&) = delete;
  void operator=(const CompileOnlyClient&) = delete;

  // A description of an xla computation to compile using CompileAheadOfTime.
  struct AotXlaComputationInstance {
    const XlaComputation* computation;
    // Inform the compiler of the expected layout for arguments.
    std::vector<const Shape*> argument_layouts;
    // Specifies the expected result layout.
    const Shape* result_layout;
  };

  // Compiles a list of xla computations for ahead-of-time execution.
  // This is intended for use in static compilation. The |options|
  // parameter describes the target for which the compiler should emit
  // code. |metadata|, if provided, is populated during compilation.
  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      absl::Span<const AotXlaComputationInstance> computations,
      const AotCompilationOptions& options,
      std::unique_ptr<AotCompilationMetadata>* metadata = nullptr);

  // Create a Hlo module config for the given program shape and arguments.
  // execution_options is optional; if not given a default is used.
  absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      absl::Span<const Shape* const> argument_shapes,
      const ExecutionOptions* execution_options);

  // Returns the size of a pointer in bytes for a given triple.
  static int64_t PointerSizeForTriple(absl::string_view triple);

 private:
  CompileOnlyService* compiler_service_;
};

}  // namespace xla

#endif  // XLA_CLIENT_COMPILE_ONLY_CLIENT_H_
