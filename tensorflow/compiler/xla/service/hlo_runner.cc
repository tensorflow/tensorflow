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

HloRunner::~HloRunner() {
  // Deallocate all the memory allocated during the tests.
  for (auto& allocation : allocations_) {
    backend().default_stream_executor()->Deallocate(&allocation);
  }
}

StatusOr<se::DeviceMemoryBase> HloRunner::Execute(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    Shape* result_shape, bool run_hlo_passes) {
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
  run_options.set_stream(&stream);
  run_options.set_allocator(backend().memory_allocator());
  run_options.set_inter_op_thread_pool(backend().inter_op_thread_pool());
  run_options.set_intra_op_thread_pool(
      backend().eigen_intra_op_thread_pool_device());

  ServiceExecutableRunOptions service_run_options(
      run_options, backend().StreamBorrower(),
      backend().inter_op_thread_pool());
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase result,
      executable->ExecuteOnStream(&service_run_options, arguments,
                                  /*hlo_execution_profile=*/nullptr));
  TF_RET_CHECK(stream.BlockHostUntilDone());

  allocations_.push_back(result);

  *result_shape = executable->result_shape();

  if (ShapeUtil::IsTuple(*result_shape)) {
    // We must record element buffers of tuples as well to avoid leaks.
    DCHECK(!ShapeUtil::IsNestedTuple(*result_shape));
    TF_ASSIGN_OR_RETURN(
        std::vector<se::DeviceMemoryBase> element_buffers,
        backend().transfer_manager()->ShallowCopyTupleFromDevice(
            backend().default_stream_executor(), result, *result_shape));

    // A tuple may contain the same buffer in more than one element. Keep track
    // of the buffers already added to avoid duplicates in allocations_.
    std::set<void*> added_opaques;
    for (auto element_buffer : element_buffers) {
      if (added_opaques.count(element_buffer.opaque()) == 0) {
        CHECK(element_buffer.opaque() != nullptr);
        added_opaques.insert(element_buffer.opaque());
        allocations_.push_back(element_buffer);
      }
    }
  }

  return result;
}

StatusOr<se::DeviceMemoryBase> HloRunner::TransferToDevice(
    const Literal& literal) {
  // Allocate memory on the device using the stream executor.
  int64 allocation_size =
      backend().transfer_manager()->GetByteSizeRequirement(literal.shape());
  se::DeviceMemoryBase allocation =
      backend().default_stream_executor()->AllocateArray<uint8>(
          allocation_size);
  allocations_.push_back(allocation);

  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      backend().default_stream_executor(), literal, &allocation));

  return allocation;
}

StatusOr<std::unique_ptr<Literal>> HloRunner::TransferFromDevice(
    const Shape& shape, se::DeviceMemoryBase device_base) {
  auto literal = MakeUnique<Literal>();
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralFromDevice(
      backend().default_stream_executor(), device_base, shape, shape,
      literal.get()));
  return std::move(literal);
}

StatusOr<std::unique_ptr<Literal>> HloRunner::ExecuteAndTransfer(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    bool run_hlo_passes) {
  Shape result_shape;
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase device_base,
      Execute(std::move(module), arguments, &result_shape, run_hlo_passes));
  return TransferFromDevice(result_shape, device_base);
}

Backend& HloRunner::backend() {
  if (!backend_) {
    backend_ = Backend::CreateDefaultBackend().ConsumeValueOrDie();
    VLOG(1) << "executing on platform " << backend().platform()->Name();
  }
  return *backend_;
}

}  // namespace xla
