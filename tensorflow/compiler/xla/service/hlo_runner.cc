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
#define EIGEN_USE_THREADS

#include "tensorflow/compiler/xla/service/hlo_runner.h"

#include <set>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::CreateModuleFromString(const tensorflow::StringPiece hlo_string,
                                  const DebugOptions& debug_options) {
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return tools::Parse(hlo_string, config);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::ReadModuleFromHloProtoFile(const std::string& filename,
                                      const DebugOptions& debug_options) {
  HloProto proto;

  const Status s =
      tensorflow::ReadBinaryProto(tensorflow::Env::Default(), filename, &proto);

  if (!s.ok()) {
    const Status s2 =
        tensorflow::ReadTextProto(tensorflow::Env::Default(), filename, &proto);
    if (!s2.ok()) {
      return Status(s2.code(), s.error_message() + "\n" + s2.error_message());
    }
  }

  TF_ASSIGN_OR_RETURN(
      HloModuleConfig config,
      HloModule::CreateModuleConfigFromProto(proto.hlo_module()));
  config.set_debug_options(debug_options);
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(proto.hlo_module(), config));
  return std::move(module);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::ReadModuleFromHloTextDumpFile(const std::string& filename,
                                         const DebugOptions& debug_options) {
  string hlo_string;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  filename, &hlo_string));
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return tools::Parse(hlo_string, config);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>> HloRunner::ReadModule(
    const std::string& filename, const DebugOptions& debug_options) {
  auto module = HloRunner::ReadModuleFromHloProtoFile(filename, debug_options);
  if (module.ok()) {
    return module;
  }
  const std::string e = module.status().error_message();
  module = HloRunner::ReadModuleFromHloTextDumpFile(filename, debug_options);
  return module.ok() ? std::move(module)
                     : Status(module.status().code(),
                              e + "\n" + module.status().error_message());
}

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct HloRunner::EigenThreadPoolWrapper {
  std::unique_ptr<EigenThreadPoolWrapper> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

HloRunner::HloRunner() {}

HloRunner::HloRunner(se::Platform* platform) {
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  backend_ = Backend::CreateBackend(backend_options).ConsumeValueOrDie();
  VLOG(1) << "Created HloRunner for platform: " << platform->Name();
}

HloRunner::~HloRunner() {}

StatusOr<std::unique_ptr<Literal>> HloRunner::ExecuteInternal(
    std::unique_ptr<HloModule> module,
    const tensorflow::gtl::ArraySlice<Literal*> arguments,
    bool run_hlo_passes) {
  if (run_hlo_passes) {
    TF_ASSIGN_OR_RETURN(
        module, backend().compiler()->RunHloPasses(
                    std::move(module), backend().default_stream_executor()));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      backend().compiler()->RunBackend(std::move(module),
                                       backend().default_stream_executor()));

  se::Stream stream(backend().default_stream_executor());
  stream.Init();

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(backend().default_device_ordinal());
  run_options.set_stream(&stream);
  run_options.set_allocator(backend().memory_allocator());
  run_options.set_inter_op_thread_pool(backend().inter_op_thread_pool());
  run_options.set_intra_op_thread_pool(
      backend().eigen_intra_op_thread_pool_device());

  ServiceExecutableRunOptions service_run_options(
      run_options, backend().StreamBorrower(),
      backend().inter_op_thread_pool());

  // Copy arguments to device.
  std::vector<std::unique_ptr<ScopedShapedBuffer>> argument_buffers;
  std::vector<ShapedBuffer*> argument_buffer_ptrs;
  for (Literal* argument : arguments) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<ScopedShapedBuffer> argument_buffer,
        backend().transfer_manager()->AllocateScopedShapedBuffer(
            argument->shape(), run_options.allocator(),
            run_options.device_ordinal()));
    TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
        stream.parent(), *argument, *argument_buffer));
    argument_buffers.push_back(std::move(argument_buffer));
    argument_buffer_ptrs.push_back(argument_buffers.back().get());
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ShapedBuffer> result,
      executable->ExecuteOnStream(&service_run_options, argument_buffer_ptrs,
                                  /*hlo_execution_profile=*/nullptr));

  // Create a ScopedShapedBuffer of the result to manage deallocation. This will
  // deallocate all the device memory when it goes out of scope.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ScopedShapedBuffer> scoped_result,
      ScopedShapedBuffer::MakeScoped(result.get(), run_options.allocator()));

  return backend().transfer_manager()->TransferLiteralFromDevice(
      stream.parent(), *scoped_result);
}

Backend& HloRunner::backend() {
  if (!backend_) {
    backend_ = Backend::CreateDefaultBackend().ConsumeValueOrDie();
    VLOG(1) << "executing on platform " << backend().platform()->Name();
  }
  return *backend_;
}

}  // namespace xla
