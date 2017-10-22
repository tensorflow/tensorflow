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

#include "tensorflow/compiler/xla/service/hlo_runner.h"

#include <set>
#include <string>
#include <utility>

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunner::ReadModuleFromHloProtoFile(const char* filename,
                                      const DebugOptions& debug_options) {
  HloProto proto;
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 filename, &proto));
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  TF_ASSIGN_OR_RETURN(auto module, HloModule::CreateFromProto(
                                       proto.hlo_module(),
                                       VersionedComputationHandle(), config));
  return std::move(module);
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
    Shape* result_shape) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      backend().compiler()->Compile(std::move(module),
                                    backend().default_stream_executor()));

  se::Stream stream(backend().default_stream_executor());
  stream.Init();

  ExecutableRunOptions run_options;
  run_options.set_stream(&stream);
  run_options.set_allocator(backend().memory_allocator());
  run_options.set_inter_op_thread_pool(backend().inter_op_thread_pool());
  run_options.set_intra_op_thread_pool(
      backend().eigen_intra_op_thread_pool_device());

  HloExecutionProfile hlo_execution_profile;
  ServiceExecutableRunOptions service_run_options(
      run_options, backend().StreamBorrower(),
      backend().inter_op_thread_pool());
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase result,
      executable->ExecuteOnStream(&service_run_options, arguments,
                                  &hlo_execution_profile));
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

se::DeviceMemoryBase HloRunner::TransferToDevice(const Literal& literal) {
  // Allocate memory on the device using the stream executor.
  int64 allocation_size =
      backend().transfer_manager()->GetByteSizeRequirement(literal.shape());
  se::DeviceMemoryBase allocation =
      backend().default_stream_executor()->AllocateArray<uint8>(
          allocation_size);
  allocations_.push_back(allocation);

  TF_CHECK_OK(backend().transfer_manager()->TransferLiteralToDevice(
      backend().default_stream_executor(), literal, &allocation));

  return allocation;
}

std::unique_ptr<Literal> HloRunner::TransferFromDevice(
    const Shape& shape, se::DeviceMemoryBase device_base) {
  auto literal = MakeUnique<Literal>();
  TF_CHECK_OK(backend().transfer_manager()->TransferLiteralFromDevice(
      backend().default_stream_executor(), device_base, shape, shape,
      literal.get()));
  return literal;
}

std::unique_ptr<Literal> HloRunner::ExecuteAndTransfer(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  Shape result_shape;
  se::DeviceMemoryBase device_base =
      Execute(std::move(module), arguments, &result_shape).ValueOrDie();
  return TransferFromDevice(result_shape, device_base);
}

template <>
std::unique_ptr<Literal> HloRunner::Execute(
    std::unique_ptr<HloModule> module,
    const tensorflow::gtl::ArraySlice<std::unique_ptr<Literal>>& literals) {
  std::vector<se::DeviceMemoryBase> arguments;
  for (const auto& literal : literals) {
    arguments.push_back(TransferToDevice(*literal));
  }
  return ExecuteAndTransfer(std::move(module), arguments);
}

template <>
std::unique_ptr<Literal> HloRunner::Execute(
    std::unique_ptr<HloModule> module,
    const tensorflow::gtl::ArraySlice<Literal*>& literals) {
  std::vector<se::DeviceMemoryBase> arguments;
  for (const auto& literal : literals) {
    arguments.push_back(TransferToDevice(*literal));
  }
  return ExecuteAndTransfer(std::move(module), arguments);
}

Backend& HloRunner::backend() {
  if (!backend_) {
    backend_ = Backend::CreateDefaultBackend().ConsumeValueOrDie();
    VLOG(1) << "executing on platform " << backend().platform()->Name();
  }
  return *backend_;
}

}  // namespace xla
