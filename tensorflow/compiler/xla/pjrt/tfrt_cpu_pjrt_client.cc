/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

#include <memory>

#define EIGEN_USE_THREADS

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace xla {

static const char kCpuPlatformName[] = "cpu";

TfrtCpuDevice::TfrtCpuDevice(int id, bool asynchronous)
    : id_(id),
      max_inflight_computations_semaphore_(/*capacity=*/asynchronous ? 32 : 1) {
}

absl::string_view TfrtCpuDevice::device_kind() const {
  return kCpuPlatformName;
}

std::string TfrtCpuDevice::DebugString() const {
  return absl::StrCat("TFRT_CPU_", id());
}

Status TfrtCpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  return TransferLiteralToInfeedOnCpu(local_hardware_id(), literal);
}

Status TfrtCpuDevice::TransferFromOutfeed(MutableBorrowingLiteral literal) {
  return TransferLiteralFromOutfeedOnCpu(local_hardware_id(), literal);
}

static int CpuDeviceCount() {
  // By default we fix the number of devices to one.  However we do let the user
  // override this behavior to help run tests on the host that run models in
  // parallel across multiple devices, e.g. pmap.
  return GetDebugOptionsFromFlags().xla_force_host_platform_device_count();
}

static StatusOr<std::vector<std::unique_ptr<TfrtCpuDevice>>> GetTfrtCpuDevices(
    bool asynchronous) {
  std::vector<std::unique_ptr<TfrtCpuDevice>> devices;
  for (int i = 0; i < CpuDeviceCount(); ++i) {
    auto device = std::make_unique<TfrtCpuDevice>(
        /*id=*/i, asynchronous);
    devices.push_back(std::move(device));
  }
  return std::move(devices);
}

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous) {
  // TODO(zhangqiaorjc): Allow users set the number of threads.
  // `num_blocking_threads=16` is picked arbitrarily for now.
  // Need at least CpuDeviceCount threads to launch one collective.
  int num_threads = std::max(DefaultThreadPoolSize(), CpuDeviceCount());
  auto host_context = std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(ERROR) << "Encountered runtime error: " << diag.message << "\n";
      },
      tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/num_threads,
          /*num_blocking_threads=*/16));

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                      GetTfrtCpuDevices(asynchronous));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtCpuClient>(
      /*process_index=*/0, std::move(devices), std::move(host_context)));
}

TfrtCpuClient::TfrtCpuClient(
    int process_index, std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
    std::unique_ptr<tfrt::HostContext> host_ctx)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      host_ctx_(std::move(host_ctx)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      eigen_intraop_pool_(new tensorflow::thread::ThreadPool(
          tensorflow::Env::Default(), "XLAEigen", DefaultThreadPoolSize())),
      eigen_intraop_device_(
          new Eigen::ThreadPoolDevice(eigen_intraop_pool_->AsEigenThreadPool(),
                                      eigen_intraop_pool_->NumThreads())) {
  for (const std::unique_ptr<TfrtCpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(id_to_device_.insert({device->id(), device.get()}).second)
        << "Duplicate device id: " << device->id();

    device->SetClient(this);
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id();
      if (idx >= addressable_devices_.size()) {
        addressable_devices_.resize(idx + 1);
      }
      CHECK(addressable_devices_[idx] == nullptr) << idx;
      addressable_devices_[idx] = device.get();
    }
  }
  for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
    CHECK(addressable_devices_[idx] != nullptr) << idx;
  }
  LOG(INFO) << "TfrtCpuClient created.";
}

StatusOr<PjRtDevice*> TfrtCpuClient::LookupDevice(int device_id) const {
  auto it = id_to_device_.find(device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         device_id);
}

StatusOr<PjRtDevice*> TfrtCpuClient::LookupAddressableDevice(
    int local_hardware_id) const {
  for (auto* device : addressable_devices_) {
    if (local_hardware_id == device->local_hardware_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_hardware_id %d",
                         local_hardware_id);
}

StatusOr<DeviceAssignment> TfrtCpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

StatusOr<std::unique_ptr<HloCostAnalysis>> TfrtCpuClient::GetHloCostAnalysis() {
  return absl::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
}

StatusOr<absl::optional<std::string>> TfrtCpuClient::ExecutableFingerprint(
    const PjRtExecutable& executable) const {
  return absl::optional<std::string>();
}

static StatusOr<std::unique_ptr<xla::Executable>> JitCompile(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  // Unoptimized HloModuleConfig.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_layouts, &execution_options,
                         execution_options.num_replicas(),
                         /*num_threads=*/absl::nullopt,
                         /*aot_options=*/nullptr));

  // Unoptimized HloModule.
  const xla::HloModuleProto& hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));
  VLOG(1) << "Unoptimized HLO module: " << hlo_module->ToString();

  // Run Hlo Passes
  cpu::CpuCompiler compiler;
  xla::Compiler::CompileOptions dummy;
  TF_ASSIGN_OR_RETURN(hlo_module,
                      compiler.RunHloPasses(std::move(hlo_module),
                                            /*stream_exec=*/nullptr, dummy));

  // Run backend.
  return compiler.RunBackend(std::move(hlo_module), /*stream_exec=*/nullptr,
                             dummy);
}

// Find the root instruction of the entry computation.
static const InstructionValueSet& GetRootValueSet(
    const BufferAssignment& assignment, const HloModule& module) {
  return assignment.dataflow_analysis().GetInstructionValueSet(
      module.entry_computation()->root_instruction());
}

// Buffer table is indexed by buffer allocation indices. The output buffer is
// made up of a subset of those buffer allocations (for tuple, it includes tuple
// index table). This helper finds the buffer allocation indices in buffer
// assignment that make up for the output buffer. It is used by
// CreateResultShapedBuffer to reconstruct the output buffer from the buffer
// table allocated by MemoryForAllocation.
static StatusOr<absl::InlinedVector<BufferAllocation::Index, 4>>
FindResultBufferAllocationIndex(const BufferAssignment& assignment,
                                const HloModule& module) {
  absl::InlinedVector<BufferAllocation::Index, 4> buffer_indices;
  const InstructionValueSet& root_value_set =
      GetRootValueSet(assignment, module);
  const Shape& result_shape = module.result_shape();
  if (!result_shape.IsTuple()) {
    // Find the buffer allocation that corresponds to the output buffer.
    const HloValueSet& sources = root_value_set.element({});
    // The points to set is unambiguous so the set should be a singleton.
    CHECK_EQ(1, sources.values().size());
    const HloValue* value_source = sources.values()[0];
    HloInstruction* src = value_source->instruction();
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        assignment.GetUniqueSlice(src, value_source->index()));
    const BufferAllocation::Index buffer_index = slice.index();
    buffer_indices.push_back(buffer_index);
    return {std::move(buffer_indices)};
  }
  buffer_indices.reserve(result_shape.tuple_shapes_size());
  for (int i = 0; i < result_shape.tuple_shapes_size(); ++i) {
    // Find the buffer allocations that corresponds to the output tuple,
    // including the tuple index table.
    const HloValueSet& sources = root_value_set.element({i});
    // The points to set is unambiguous so the set should be a singleton.
    CHECK_EQ(1, sources.values().size());
    const HloValue* value_source = sources.values()[0];
    HloInstruction* src = value_source->instruction();
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        assignment.GetUniqueSlice(src, value_source->index()));
    const BufferAllocation::Index buffer_index = slice.index();
    buffer_indices.push_back(buffer_index);
  }
  return {std::move(buffer_indices)};
}

StatusOr<std::unique_ptr<PjRtExecutable>> TfrtCpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuClient::Compile");
  ExecutableBuildOptions& build_options = options.executable_build_options;

  int num_replicas;
  int num_partitions;
  std::shared_ptr<DeviceAssignment> device_assignment;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options.compile_portable_executable, &options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  std::vector<const Shape*> argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation, &LayoutUtil::GetWithDefaultLayout, options.argument_layouts,
      &options.executable_build_options, &argument_layout_pointers));

  std::vector<PjRtExecutable::LogicalDeviceIds> addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        int device_id = (*device_assignment)(replica, partition);
        TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
        if (device->process_index() != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        PjRtExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }
    if (addressable_devices.empty()) {
      return InvalidArgument(
          "Device assignment (%s) does not have any local devices.",
          device_assignment->ToString());
    }

    if (build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id());
    }
  }

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> cpu_executable,
                      JitCompile(computation, argument_layout_pointers,
                                 build_options, execution_options));
  auto cpu_executable_ptr =
      down_cast<cpu::CpuExecutable*>(cpu_executable.get());

  // `buffer_table[result_slice.index()]` points to result buffer:
  // If output is a tuple, it points to the buffer index table.
  // If output is a non-tuple, it points to the buffer itself.
  TF_ASSIGN_OR_RETURN(
      const BufferAllocation::Slice result_slice,
      cpu_executable_ptr->buffer_assignment().GetUniqueTopLevelOutputSlice());

  // `result_buffer_indices` has the buffer allocation indices that make up the
  // output buffer (could be tuple).
  TF_ASSIGN_OR_RETURN(
      auto result_buffer_indices,
      FindResultBufferAllocationIndex(cpu_executable_ptr->buffer_assignment(),
                                      cpu_executable->module()));

  auto executable = std::make_unique<TfrtCpuExecutable>(
      num_replicas, num_partitions, std::move(device_assignment),
      options.parameter_is_tupled_arguments, std::move(cpu_executable),
      result_slice.index(), std::move(result_buffer_indices),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);
  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));

  return std::unique_ptr<PjRtExecutable>(std::move(executable));
}

TfrtCpuExecutable::TfrtCpuExecutable(
    int num_replicas, int num_partitions,
    std::shared_ptr<DeviceAssignment> device_assignment,
    bool parameter_is_tupled_arguments,
    std::unique_ptr<Executable> cpu_executable,
    BufferAllocation::Index result_buffer_index,
    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, TfrtCpuClient* client)
    : client_(client),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      device_assignment_(std::move(device_assignment)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      cpu_executable_(std::move(cpu_executable)),
      result_buffer_index_(result_buffer_index),
      result_buffer_indices_(std::move(result_buffer_indices)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {}

void TfrtCpuExecutable::Delete() {}

StatusOr<absl::optional<std::string>> TfrtCpuExecutable::Fingerprint() const {
  return absl::optional<std::string>();
}

Status TfrtCpuExecutable::SetUpDonation(bool tuple_inputs) {
  TF_ASSIGN_OR_RETURN(parameters_that_must_be_donated_,
                      GetParametersThatMustBeDonated(
                          *cpu_executable_->shared_module(), tuple_inputs));
  return Status::OK();
}

bool TfrtCpuExecutable::MustDonateParameter(int parameter) const {
  return parameters_that_must_be_donated_.contains(parameter);
}

}  // namespace xla
