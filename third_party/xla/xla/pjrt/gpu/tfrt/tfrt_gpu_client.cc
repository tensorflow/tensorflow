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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace {

static std::string get_platform_version(xla::LocalClient* xla_client) {
  const stream_executor::DeviceDescription& device =
      xla_client->backend().default_stream_executor()->GetDeviceDescription();
  if (std::holds_alternative<stream_executor::RocmComputeCapability>(
          device.gpu_compute_capability())) {
    return absl::StrCat("rocm ", device.runtime_version());
  }
  if (std::holds_alternative<stream_executor::CudaComputeCapability>(
          device.gpu_compute_capability())) {
    return absl::StrCat("cuda ", device.runtime_version());
  }
  return "<unknown>";
}

bool IsAllZeros(const DeviceAssignment& assignment) {
  return std::all_of(
      assignment.begin(), assignment.end(),
      [](const DeviceAssignment::value_type& v) { return v == 0; });
}

}  // namespace

TfrtGpuMemorySpace::TfrtGpuMemorySpace(int id, PjRtDevice* device,
                                       absl::string_view kind, int kind_id)
    : id_(id), device_(device), kind_(kind), kind_id_(kind_id) {
  DCHECK(device_ != nullptr && device_->client() != nullptr);
  to_string_ = absl::StrCat("MEMORY_SPACE_", id_);
  debug_string_ = absl::StrFormat("TfrtGpuMemory(id=%i, device=%s)", id_,
                                  device_->DebugString());
}

const int TfrtGpuDeviceMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(TfrtGpuDeviceMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

TfrtGpuDeviceMemorySpace::TfrtGpuDeviceMemorySpace(int id, PjRtDevice* device)
    : TfrtGpuMemorySpace(id, device, kKind, kKindId) {}

TfrtGpuDevice::TfrtGpuDevice(Options&& options)
    : id_(options.id),
      local_device_id_(options.local_device_id),
      local_hardware_id_(options.local_hardware_id),
      stream_pool_(options.executor, options.stream_capacity),
      compute_stream_pool_(options.executor, options.max_inflight_computations),
      description_(options.id, options.platform_version) {
  description_.SetDebugString(absl::StrCat("TFRT_GPU_", id_));
  description_.SetToString(absl::StrCat("GpuDevice(id=", id_, ")"));
}

absl::Status TfrtGpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  return Unimplemented("TransferToInfeed");
}

absl::Status TfrtGpuDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  return Unimplemented("TransferFromOutfeed");
}

void TfrtGpuDevice::AttachMemorySpace(PjRtMemorySpace* memory_space,
                                      bool is_default) {
  CHECK(memory_space != nullptr);
  CHECK(client_ == memory_space->client()) << absl::StrFormat(
      "Could not attach a PjRtStreamExecutorDevice to a PjRtMemorySpace owned "
      "by a different client, the device's client: %s, the memory space's "
      "client: %s.",
      client_->platform_name(), memory_space->client()->platform_name());

  memory_spaces_.push_back(memory_space);
  memory_spaces_by_kind_id_.emplace(memory_space->kind_id(), memory_space);
  if (is_default) {
    CHECK(default_memory_space_ == nullptr)
        << "Default memory space already set to "
        << default_memory_space_->DebugString() << ".";
    default_memory_space_ = memory_space;
  }
}

absl::Span<PjRtMemorySpace* const> TfrtGpuDevice::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind_id(
    int id) const {
  auto it = memory_spaces_by_kind_id_.find(id);
  if (it == memory_spaces_by_kind_id_.end()) {
    return absl::InternalError(
        absl::StrCat("No memory space found (kind_id: ", id, ")"));
  }
  return it->second;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind(
    absl::string_view kind) const {
  auto it = absl::c_find_if(memory_spaces_, [kind](PjRtMemorySpace* ms) {
    return ms->kind() == kind;
  });
  if (it != memory_spaces_.end()) {
    return *it;
  }
  return absl::InternalError(
      absl::StrCat("No memory space found (kind: ", kind, ")"));
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::default_memory_space() const {
  if (default_memory_space_ == nullptr) {
    return absl::InternalError(
        "No default memory space is set for this device.");
  }
  return default_memory_space_;
}

TfrtGpuClient::TfrtGpuClient(
    int process_index, xla::LocalClient* xla_client,
    std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    std::shared_ptr<const GpuTopology> gpu_topology)
    : process_index_(process_index),
      xla_client_(CHECK_NOTNULL(xla_client)),
      platform_version_(get_platform_version(xla_client)),
      owned_devices_(std::move(devices)),
      compile_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_compile_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()))) {
  for (const std::unique_ptr<TfrtGpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(
        id_to_device_.emplace(device->global_device_id(), device.get()).second)
        << "Duplicate device id: " << device->id();

    device->SetClient(this);
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id().value();
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

  for (auto* device : addressable_devices()) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    // Initialize the default memory space.
    const int global_device_id = gpu_device->global_device_id().value();
    auto memory_space =
        std::make_unique<TfrtGpuDeviceMemorySpace>(global_device_id, device);
    gpu_device->AttachMemorySpace(memory_space.get(), /*is_default=*/true);
    memory_spaces_.push_back(memory_space.get());
    owned_memory_spaces_.push_back(std::move(memory_space));
  }
  const int basePinnedId = device_count();
  for (auto* device : addressable_devices()) {
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    const int global_device_id = gpu_device->global_device_id().value();
    auto pinned = std::make_unique<PinnedHostMemorySpace>(
        basePinnedId + global_device_id, device);
    gpu_device->AttachMemorySpace(pinned.get());
    memory_spaces_.push_back(pinned.get());
    owned_memory_spaces_.push_back(std::move(pinned));
  }

  LOG(INFO) << "TfrtGpuClient created.";
}

absl::Span<PjRtMemorySpace* const> TfrtGpuClient::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtGpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = xla_client_,
       allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return local_client->backend()
            .transfer_manager()
            ->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInternal(computation, argument_layout_pointers,
                         /* layout_canonicalization_callback = */ nullptr,
                         options);

  // TODO(b/382117736): Record free gpu memory.
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/gpu/se_gpu_pjrt_client.cc#L792-L809
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::Compile");
  VLOG(1) << "TfrtGpuClient::Compile";
  if (key_value_store().has_value() &&
      !options.executable_build_options.key_value_store()) {
    options.executable_build_options.set_key_value_store(*key_value_store());
  }
  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras, GetExecutableExtras(&options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  // It is important to set the canonicalization callback after creating
  // a copy of the options so that the executable's options remain without
  // the callback - the callback would break the executable's serializability.
  if (layout_canonicalization_callback) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      xla_client_->Compile(computation, argument_layout_pointers,
                           options.executable_build_options));

  auto executable = std::make_unique<TfrtGpuExecutable>(
      std::move(local_executables), options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));
  const auto& ex_options = options.executable_build_options;
  if (ex_options.has_debug_options() &&
      ex_options.debug_options().xla_gpu_dump_hlo_unoptimized_snapshots()) {
    executable->SetInputHloSnapshotBits(
        computation.proto(), options.executable_build_options.debug_options());
  }
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtGpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  XlaComputation xla_computation;
  const ExecutableBuildOptions& exec_build_options =
      options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, exec_build_options.use_shardy_partitioner()));

  // TODO(b/382117736): Add support for LayoutModesToXlaShapes
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/pjrt_stream_executor_client.cc#L3538-L3586
  return Compile(xla_computation, options);
}

absl::StatusOr<TfrtGpuClient::ExecutableExtras>
TfrtGpuClient::GetExecutableExtras(CompileOptions* options) {
  ExecutableExtras extras;
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  ExecutableBuildOptions& build_options = options->executable_build_options;
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(compile_thread_pool_.get());
  }

  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(allocator_);
  }

  auto layout_callback = [local_client = xla_client_,
                          options](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    ExecutableBuildOptions build_options = options->executable_build_options;
    std::vector<const Shape*> argument_layout_pointers;
    std::optional<std::vector<Shape>> argument_layouts =
        options->argument_layouts;
    Shape result_layout;
    const bool allow_auto_layout =
        build_options.has_debug_options() &&
        build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
    TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
        XlaComputation(module.ToProto()),
        [local_client,
         allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
          if (allow_auto_layout && !shape.has_layout()) {
            return shape;
          }
          return local_client->backend()
              .transfer_manager()
              ->ChooseCompactLayoutForShape(shape);
        },
        argument_layouts, &build_options, &argument_layout_pointers));
    result_layout = *build_options.result_layout();
    return std::make_pair(*argument_layouts, result_layout);
  };

  build_options.set_layout_canonicalization_callback(layout_callback);

  int num_replicas;
  int num_partitions;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options->compile_portable_executable, &options->executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  // Find devices that are addressable by this client/task.
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    absl::flat_hash_set<int> all_process_indices;
    std::optional<int> this_process_index;
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        int64_t device_id = (*device_assignment)(replica, partition);
        PjRtGlobalDeviceId global_device_id(device_id);

        TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                            LookupDevice(global_device_id));
        all_process_indices.insert(device->process_index());
        if (device->process_index() != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        if (!this_process_index.has_value()) {
          this_process_index = all_process_indices.size() - 1;
        }
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
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
          addressable_devices.front()->local_hardware_id().value());
    }

    build_options.set_process_index(*this_process_index);
    build_options.set_process_count(all_process_indices.size());
  }
  return extras;
}

static absl::StatusOr<std::vector<std::unique_ptr<TfrtGpuDevice>>>
GetTfrtGpuDevices(LocalClient* xla_client) {
  std::vector<std::unique_ptr<TfrtGpuDevice>> devices;
  int i = 0;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    // TODO(b/382117736): allow GPU allocator parameters to be configurable.
    TF_ASSIGN_OR_RETURN(auto allocator,
                        CreateBFCAllocator(executor, /*memory_fraction=*/0.9,
                                           /*preallocate=*/true, std::nullopt));

    TfrtGpuDevice::Options options;
    options.id = i;
    options.local_device_id = PjRtLocalDeviceId(i);
    options.local_hardware_id = PjRtLocalHardwareId(i);
    options.executor = executor;
    options.allocator = std::move(allocator);
    options.stream_capacity = 4;
    options.max_inflight_computations = 1;
    const se::Platform* platform = executor->GetPlatform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(options.local_hardware_id.value()));
    options.platform_version = desc->name();

    auto device = std::make_unique<TfrtGpuDevice>(std::move(options));
    devices.push_back(std::move(device));
    ++i;
  }
  return std::move(devices);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options) {
  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  EnablePeerAccess(xla_client->backend().stream_executors());
  std::unique_ptr<tsl::Allocator> host_memory_allocator;
  if (!xla_client->backend().stream_executors().empty()) {
    TF_ASSIGN_OR_RETURN(
        host_memory_allocator,
        GetGpuHostAllocator(xla_client->backend().stream_executors().front()));
  }
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                      GetTfrtGpuDevices(xla_client));

  GpuTopologyProto gpu_topology_proto;
  for (const auto& device : devices) {
    if (gpu_topology_proto.platform_version().empty()) {
      gpu_topology_proto.set_platform_version(
          std::string(device->device_kind()));
    }
    gpu_topology_proto.add_device_ids(device->id());
  }

  // TODO(b/382117736): Support multi-host
  gpu_topology_proto.set_num_slices(1);
  gpu_topology_proto.set_num_hosts_per_slice(1);
  gpu_topology_proto.set_num_devices_per_host(devices.size());

  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(gpu_topology_proto));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtGpuClient>(
      /*process_index=*/0, xla_client, std::move(devices),
      std::move(host_memory_allocator), gpu_topology));
}

TfrtGpuExecutable::TfrtGpuExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    bool parameter_is_tupled_arguments,
    std::shared_ptr<DeviceAssignment> device_assignment,
    CompileOptions compile_options,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, TfrtGpuClient* client)
    : client_(client),
      device_assignment_(std::move(device_assignment)),
      compile_options_(std::move(compile_options)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {
  executables_.reserve(executables.size());
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(fingerprint_);
  for (auto& executable : executables) {
    const auto& computation_layout =
        executable->executable()->module().entry_computation_layout();
    std::vector<Shape> parameter_shapes;
    parameter_shapes.reserve(computation_layout.parameter_count());
    for (int i = 0; i < computation_layout.parameter_count(); ++i) {
      // TODO(b/382117736): Convert to device shape when we have transfer
      // manager.
      // parameter_shapes.push_back(transfer_manager->HostShapeToDeviceShape(
      // computation_layout.parameter_shape(i)));
      parameter_shapes.push_back(computation_layout.parameter_shape(i));
    }
    fingerprint = tsl::FingerprintCat128(
        fingerprint,
        tsl::Fingerprint128(executable->executable()->module().ToString()));
    executables_.emplace_back(std::move(executable));
  }
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);

  int num_partitions;
  if (device_assignment_ == nullptr) {
    // This must go after `executables_` is initialized.
    VLOG(3) << "PjRtStreamExecutorLoadedExecutable portable single-core";
    num_partitions = 1;
    CHECK(addressable_devices_.empty());
  } else {
    // This must go after `executables_` is initialized.
    VLOG(3) << "PjRtStreamExecutorLoadedExecutable device_assignment:\n"
            << device_assignment_->ToString();
    CHECK_GE(addressable_devices_.size(), 1) << device_assignment_->ToString();

    if ((device_assignment_->replica_count() > 1 ||
         device_assignment_->computation_count() > 1) &&
        IsAllZeros(*device_assignment_)) {
      // This code path should only be triggered when we intentionally compile
      // an HLO without having enough devices to actually run it. See the
      // "--run=false" option in
      // tensorflow/compiler/xla/tools/multihost_hlo_runner/hlo_runner_main.cc.
      // That will help us debug the XLA compiler locally.
      LOG(INFO)
          << "A workaround is in effect to allow compiling multi-device "
             "HLOs on machines with fewer devices. Don't run this executable.";
    } else {
      CHECK_LE(addressable_devices_.size(), client_->addressable_device_count())
          << "Inconsistent local device count.";
    }

    num_partitions = device_assignment_->computation_count();
  }

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }
}

absl::string_view TfrtGpuExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    return "<unknown executable>";
  }
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
TfrtGpuExecutable::GetHloModules() const {
  std::vector<std::shared_ptr<HloModule>> modules;
  modules.reserve(executables_.size());
  for (const auto& local_exec : executables_) {
    if (!local_exec->executable()->has_module()) {
      return InvalidArgument("Executable does not have HLO modules.");
    }
    modules.push_back(local_exec->executable()->shared_module());
  }
  return std::move(modules);
}

namespace {

absl::StatusOr<absl::string_view> MemoryKindFromSimpleShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.has_layout()) {
    return default_memory_kind;
  }
  switch (shape.layout().memory_space()) {
    case Layout::kHostMemorySpace:
      return PinnedHostMemorySpace::kKind;
    case Layout::kGenericFastMemorySpace:
    case Layout::kDefaultMemorySpace:
      return default_memory_kind;
    default:
      return InvalidArgument("Unexpected memory space %d in output layout",
                             shape.layout().memory_space());
  }
}

absl::StatusOr<std::vector<absl::string_view>> MemoryKindsFromShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.IsTuple()) {
    TF_ASSIGN_OR_RETURN(absl::string_view memory_kind,
                        MemoryKindFromSimpleShape(shape, default_memory_kind));
    return {{memory_kind}};
  }
  std::vector<absl::string_view> result;
  result.reserve(shape.tuple_shapes_size());
  for (const auto& element_shape : shape.tuple_shapes()) {
    TF_ASSIGN_OR_RETURN(
        absl::string_view element_memory_kind,
        MemoryKindFromSimpleShape(element_shape, default_memory_kind));
    result.push_back(element_memory_kind);
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
TfrtGpuExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  if (addressable_devices().empty()) {
    return Unimplemented(
        "GetOutputMemoryKinds is not supported when there are no addressable "
        "devices in TfrtGpuExecutable.");
  }
  TF_ASSIGN_OR_RETURN(PjRtMemorySpace * default_memory_space,
                      addressable_devices()[0]->default_memory_space());
  std::vector<std::vector<absl::string_view>> out;
  out.reserve(shapes.size());
  for (const auto& shape : shapes) {
    TF_ASSIGN_OR_RETURN(
        std::vector<absl::string_view> memory_kind,
        MemoryKindsFromShape(shape, default_memory_space->kind()));
    out.push_back(memory_kind);
  }
  return out;
}

absl::Status TfrtGpuExecutable::SetUpDonation(bool tuple_inputs) {
  parameters_that_must_be_donated_.reserve(executables_.size());
  for (auto& executable : executables_) {
    TF_ASSIGN_OR_RETURN(std::vector<int> parameters_to_donate,
                        ComputeParametersThatMustBeDonated(
                            executable->executable()->module(), tuple_inputs));
    parameters_that_must_be_donated_.emplace_back(
        std::move(parameters_to_donate));
  }
  return absl::OkStatus();
}

}  // namespace xla
