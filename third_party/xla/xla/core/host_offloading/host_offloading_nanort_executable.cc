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

#include "xla/core/host_offloading/host_offloading_nanort_executable.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/nanort/nanort_client.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/core/host_offloading/host_offloading_buffer.h"
#include "xla/core/host_offloading/host_offloading_executable.h"
#include "xla/core/host_offloading/host_offloading_layout_analysis.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/utils.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

// An upper bound on the number of threads to use for intra-op parallelism. It
// is nearly impossible to utilize efficiently more than 256 threads for compute
// intensive operations that are supposed to run inside the intra-op threadpool.
static const size_t kMaxIntraOpThreads = 256;

static tsl::ThreadOptions GetIntraOpThreadOptions() {
  tsl::ThreadOptions thread_options;
  // On Mac OS the default stack size is 512KiB, which is too small for some
  // BLAS and LAPACK functions (https://github.com/google/jax/issues/20428).
  // On Linux we also observed that 2MB wasn't enough to run some OpenBLAS
  // functions.
  thread_options.stack_size = 8 * 1024 * 1024;
  return thread_options;
}

static size_t GetIntraOpThreadPoolSize() {
  // By default we fix the number of devices to one.  However we do let the user
  // override this behavior to help run tests on the host that run models in
  // parallel across multiple devices, e.g. pmap.
  int cpu_device_count =
      GetDebugOptionsFromFlags().xla_force_host_platform_device_count();
  size_t num_threads = std::max(DefaultThreadPoolSize(), cpu_device_count);
  return std::min(num_threads, kMaxIntraOpThreads);
}

static tsl::thread::ThreadPool& GetIntraOpThreadPool() {
  static absl::NoDestructor<tsl::thread::ThreadPool> intra_op_thread_pool(
      tsl::Env::Default(), GetIntraOpThreadOptions(),
      "host-offloading-intra-op", GetIntraOpThreadPoolSize());
  return *intra_op_thread_pool;
}

HostOffloadingNanoRtExecutable::HostOffloadingNanoRtExecutable(
    std::string name, ProgramShape program_shape,
    HloInputOutputAliasConfig alias_config,
    std::unique_ptr<xla::cpu::NanoRtExecutable> executable,
    bool needs_layout_conversion,
    std::shared_ptr<DeviceAssignment> device_assignment)
    : name_(std::move(name)),
      program_shape_(std::move(program_shape)),
      alias_config_(std::move(alias_config)),
      executable_(std::move(executable)),
      needs_layout_conversion_(needs_layout_conversion),
      device_assignment_(std::move(device_assignment)),
      intra_op_device_(GetIntraOpThreadPool().AsEigenThreadPool(),
                       GetIntraOpThreadPool().NumThreads()) {}

namespace {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

// A mutex for a global NANORT CPU client initialization.
ABSL_CONST_INIT absl::Mutex host_offloading_client_mutex(absl::kConstInit);

// Returns a global NANORT CPU client for host offloading computations.
absl::StatusOr<xla::cpu::NanoRtClient*> GetHostOffloadingNanoRtClient() {
  static xla::cpu::NanoRtClient* client = nullptr;

  absl::MutexLock lock(&host_offloading_client_mutex);
  if (client != nullptr) {
    return client;
  }

  VLOG(1) << "Create host offloading NanoRt client for a current process";
  client = new xla::cpu::NanoRtClient();
  return client;
}

}  // namespace

absl::StatusOr<std::unique_ptr<HostOffloadingNanoRtExecutable>>
HostOffloadingNanoRtExecutable::LoadFromProto(
    const HostOffloadingExecutableProto& proto) {
  TF_RET_CHECK(proto.executable_type() ==
               HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT);

  VLOG(3) << "Load NanoRt host offloading executable: name="
          << proto.hlo_module().name();

  TraceMe trace([&] {
    return TraceMeEncode("HostOffloadingNanoRtExecutable::LoadFromProto",
                         {{"name", proto.hlo_module().name()}});
  });

  // We keep program shape and alias config of the original HLO module and not
  // the destination-passing-style module with extra output parameters.
  TF_ASSIGN_OR_RETURN(
      ProgramShape program_shape,
      ProgramShape::FromProto(proto.hlo_module().host_program_shape()));
  TF_ASSIGN_OR_RETURN(
      auto alias_config,
      HloInputOutputAliasConfig::CreateFromProto(
          program_shape.result(), proto.hlo_module().input_output_alias()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      HloModule::CreateFromProto(
                          proto.hlo_module(), HloModuleConfig(program_shape)));

  XlaComputation computation;
  computation = XlaComputation(proto.hlo_module());

  TF_ASSIGN_OR_RETURN(
      bool needs_layout_conversion,
      HostOffloadingLayoutAnalysis::NeedsLayoutConversion(hlo_module.get()));

  TF_ASSIGN_OR_RETURN(xla::cpu::NanoRtClient * client,
                      GetHostOffloadingNanoRtClient());

  // TODO(basioli): Add support for compile options.
  CompileOptions compile_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::cpu::NanoRtExecutable> executable,
                      client->Compile(computation));

  std::shared_ptr<DeviceAssignment> device_assignment;
  int num_replicas;
  int num_partitions;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      compile_options.compile_portable_executable,
      &compile_options.executable_build_options,
      [](int num_replicas, int num_partitions) {
        ComputationPlacer computation_placer;
        return computation_placer.AssignDevices(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  return absl::WrapUnique(new HostOffloadingNanoRtExecutable(
      proto.hlo_module().name(),
      executable->program_shape() ? *executable->program_shape()
                                  : program_shape,
      std::move(alias_config), std::move(executable), needs_layout_conversion,
      std::move(device_assignment)));
}

tsl::AsyncValueRef<HostOffloadingExecutable::ExecuteEvent>
HostOffloadingNanoRtExecutable::Execute(
    absl::Span<const ShapeTree<HostOffloadingBuffer>> parameters,
    const xla::ShapeTree<HostOffloadingBuffer>& result,
    const ExecuteOptions& execute_options) {
  VLOG(3) << "Execute NanoRt host offloading executable: name=" << name_;

  TraceMe trace([&] {
    return TraceMeEncode(
        "HostOffloadingNanoRtExecutable::Execute",
        {{"executable", absl::StrFormat("%s (device %d)", name_,
                                        execute_options.device_index)},
         {"launch_id", execute_options.launch_id}});
  });

  // Convert parameters to NanoRt arguments.
  absl::InlinedVector<xla::cpu::NanoRtExecutable::Argument, 4> arguments;

  auto add_argument = [&](const Shape& shape,
                          const HostOffloadingBuffer& buffer) {
    DCHECK(shape.IsArray()) << "Buffer shape must be an array";
    const size_t num_bytes = ShapeUtil::ByteSizeOf(shape);
    arguments.emplace_back(buffer.opaque_base(), num_bytes);
  };

  for (size_t i = 0; i < parameters.size(); ++i) {
    const ShapeTree<HostOffloadingBuffer>& parameter = parameters[i];
    for (const auto& [index, buffer] : parameter.leaves()) {
      auto shape = ShapeUtil::GetSubshape(parameter.shape(), index);
      add_argument(shape, buffer);
    }
  }

  absl::InlinedVector<xla::cpu::NanoRtExecutable::Result, 4> nanort_results;

  for (const auto& [index, buffer] : result.leaves()) {
    auto shape = ShapeUtil::GetSubshape(result.shape(), index);
    const size_t num_bytes = ShapeUtil::ByteSizeOf(shape);
    nanort_results.emplace_back(buffer.opaque_base(), num_bytes);
  }

  xla::cpu::NanoRtExecutable::ExecuteOptions nanort_execute_options;
  if (execute_options.context != nullptr) {
    nanort_execute_options.set_ffi_context(
        &execute_options.context->ffi_context());
  }
  nanort_execute_options.set_intra_op_thread_pool(&intra_op_device_);
  nanort_execute_options.set_launch_id(execute_options.launch_id);

  // We assume that for host offloading computation we have a single device.
  int32_t device_id = 0;
  nanort_execute_options.set_local_device_id(LocalDeviceId(device_id));
  nanort_execute_options.set_global_device_id(GlobalDeviceId(device_id));

  nanort_execute_options.set_device_assignment(device_assignment_.get());

  auto temp_buffer =
      std::make_unique<xla::cpu::NanoRtExecutable::ManagedTemp<128>>(
          executable_->temp_buffer_size());

  auto execute_event = executable_->Execute(
      arguments, nanort_results, *temp_buffer, nanort_execute_options);

  // Avoid creating a callback if the computation is already done.
  if (ABSL_PREDICT_TRUE(execute_event.IsAvailable())) {
    return execute_event;
  }

  // Keep arguments to Execute alive until the computation is done.
  execute_event.AndThen([arguments = std::move(arguments),
                         nanort_results = std::move(nanort_results),
                         temp_buffer = std::move(temp_buffer),
                         nanort_execute_options = std::move(
                             nanort_execute_options)](absl::Status status) {});

  return execute_event;
}

}  // namespace xla
