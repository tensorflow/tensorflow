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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPILE_ONLY_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPILE_ONLY_SERVICE_H_

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// An XLA Service specialization for ahead-of-time compilation.  This only
// instantiates a Compiler object for the relevant platform; it does not
// instantiate or require an execution backend.
class CompileOnlyService : public Service {
 public:
  // Factory for creating a CompileOnlyService. The parameter platform is the
  // platform that the service should target. If platform is null then the
  // default platform is used.
  static StatusOr<std::unique_ptr<CompileOnlyService>> NewService(
      se::Platform* platform);
  static StatusOr<std::unique_ptr<CompileOnlyService>> NewService(
      const ServiceOptions& options);

  // A description of a xla computation to compile using CompileAheadOfTime.
  struct AotXlaComputationInstance {
    HloModuleProto computation;
    std::vector<const Shape*> argument_layouts;
    const Shape* result_layout = nullptr;
  };

  // Compiles a list of xla computations for ahead-of-time execution.  This is
  // intended for use in static compilation.  See
  // |CompileOnlyClient::CompileAheadOfTime| for additional details.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      const tensorflow::gtl::ArraySlice<AotXlaComputationInstance> computations,
      const AotCompilationOptions& options);

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      const tensorflow::gtl::ArraySlice<AotXlaComputationInstance> computations,
      const AotCompilationOptions& options,
      std::unique_ptr<AotCompilationMetadata>* metadata);

  Status GetDeviceHandles(const GetDeviceHandlesRequest* arg,
                          GetDeviceHandlesResponse* result) override {
    return Unimplemented("CompileOnlyService does not support devices.");
  }
  Status WaitForExecution(const WaitForExecutionRequest* arg,
                          WaitForExecutionResponse* result) override {
    return Unimplemented("CompileOnlyService does not support execution.");
  }
  Status TransferToServer(const TransferToServerRequest* arg,
                          TransferToServerResponse* result) override {
    return Unimplemented(
        "CompileOnlyService does not support device data transfers.");
  }
  Status TransferToInfeed(const TransferToInfeedRequest* arg,
                          TransferToInfeedResponse* result) override {
    return Unimplemented(
        "CompileOnlyService does not support device data transfers.");
  }
  Status TransferFromOutfeed(const TransferFromOutfeedRequest* arg,
                             TransferFromOutfeedResponse* result) override {
    return Unimplemented(
        "CompileOnlyService does not support device data transfers.");
  }
  Status ResetDevice(const ResetDeviceRequest* arg,
                     ResetDeviceResponse* result) override {
    return Unimplemented("CompileOnlyService does not support devices.");
  }

 private:
  explicit CompileOnlyService(const ServiceOptions& options,
                              Compiler* compiler);
  CompileOnlyService(const CompileOnlyService&) = delete;
  void operator=(const CompileOnlyService&) = delete;

  // The compiler for the target platform.  This is included in place of
  // the Service::execute_backend_'s compiler, since execute_backend_ is a
  // nullptr in CompileOnlyService.
  Compiler* compiler_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILE_ONLY_SERVICE_H_
