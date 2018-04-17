/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_COMPILE_ONLY_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_COMPILE_ONLY_CLIENT_H_

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/service/compile_only_service.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

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

  // A description of a computation to compile using CompileAheadOfTime.
  struct AotComputationInstance {
    const Computation* computation;
    // Inform the compiler of the expected layout for arguments.
    std::vector<const Shape*> argument_layouts;
    // Specifies the expected result layout.
    const Shape* result_layout;
  };

  // Compiles a list of computations for ahead-of-time execution.  This is
  // intended for use in static compilation. The |options| parameter describes
  // the target for which the compiler should emit code.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      const tensorflow::gtl::ArraySlice<AotComputationInstance> computations,
      const AotCompilationOptions& options);

  // A description of an xla computation to compile using CompileAheadOfTime.
  //
  // TODO(b/74197823): This is a part of a NOT YET ready refactor.
  struct AotXlaComputationInstance {
    const XlaComputation* computation;
    // Inform the compiler of the expected layout for arguments.
    std::vector<const Shape*> argument_layouts;
    // Specifies the expected result layout.
    const Shape* result_layout;
  };

  // Compiles a list of xla computations for ahead-of-time execution.  This is
  // intended for use in static compilation. The |options| parameter describes
  // the target for which the compiler should emit code.
  //
  // TODO(b/74197823): This is a part of a NOT YET ready refactor.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      const tensorflow::gtl::ArraySlice<AotXlaComputationInstance> computations,
      const AotCompilationOptions& options);

  // Returns the size of a pointer in bytes for a given triple.
  static int64 PointerSizeForTriple(tensorflow::StringPiece triple);

 private:
  CompileOnlyService* compiler_service_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_COMPILE_ONLY_CLIENT_H_
