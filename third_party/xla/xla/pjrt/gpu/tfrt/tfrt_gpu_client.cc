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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "riegeli/bytes/string_reader.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/maybe_owning.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/dump/dump.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_async_host_to_device_transfer_manager.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_executable.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/gpu/tfrt/utils.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/unbounded_work_queue.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/tsl/platform/status_macros.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#if defined(PLATFORM_WINDOWS)
// Required to build successfully with Mingw
#undef CreateEvent
#endif

namespace xla {

namespace {

class UnboundedAsyncWorkRunner : public AsyncWorkRunner {
 public:
  explicit UnboundedAsyncWorkRunner(const std::string& name)
      : queue_(tsl::Env::Default(), name, {/*stack_size=*/512 * 1024}) {}

  void Schedule(absl::AnyInvocable<void() &&> work) override {
    // TSL TheadPool expects std::function that must be copyable, so we are
    // forced to do a little bit of manual memory management here.
    queue_.Schedule(
        [ptr = new absl::AnyInvocable<void() &&>(std::move(work))]() {
          std::move (*ptr)();
          delete ptr;
        });
  }

  void ScheduleWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
      absl::AnyInvocable<void() &&> work) override {
    tsl::RunWhenReady(values, [this, work = std::move(work)]() mutable {
      Schedule([work = std::move(work)]() mutable { std::move(work)(); });
    });
  }

 private:
  tsl::UnboundedWorkQueue queue_;
};

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

namespace {

std::shared_ptr<HostMemoryAllocator> CreateHostMemoryAllocator(
    TfrtGpuClient* client, HostMemoryAllocator::Factory factory) {
  if (factory == nullptr) {
    return nullptr;
  }

  HostMemoryAllocator::Options allocator_options;
  allocator_options.alignment = tsl::Allocator::kAllocatorAlignment;
  allocator_options.map_fn = [client](void* data, size_t size) {
    return client->DmaMap(data, size);
  };
  allocator_options.unmap_fn = [client](void* data) {
    return client->DmaUnmap(data);
  };
  return factory(std::move(allocator_options)).value();
}

}  // namespace

TfrtGpuClient::TfrtGpuClient(
    std::string platform_name, int process_index, xla::LocalClient* xla_client,
    std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
    bool should_stage_host_to_device_transfers,
    bool abort_collectives_on_failure,
    MaybeOwning<se::DeviceAddressAllocator> allocator,
    HostMemoryAllocator::Factory host_memory_allocator_factory,
    std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    std::shared_ptr<const GpuTopology> gpu_topology)
    : process_index_(process_index),
      platform_name_(std::move(platform_name)),
      xla_client_(CHECK_NOTNULL(xla_client)),
      should_stage_host_to_device_transfers_(
          should_stage_host_to_device_transfers),
      abort_collectives_on_failure_(abort_collectives_on_failure),
      allocator_(std::move(allocator)),
      host_memory_allocator_(CreateHostMemoryAllocator(
          this, std::move(host_memory_allocator_factory))),
      devices_(InitializeDevices(this, devices)),
      id_to_device_(GetIdToDeviceMap(devices)),
      addressable_devices_(GetAddressableDevicePointers(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      owned_memory_spaces_(
          InitializeMemorySpaces(devices.size(), addressable_devices_)),
      memory_spaces_(GetMemorySpacePointers(owned_memory_spaces_)),
      gpu_run_options_(std::move(gpu_run_options)),
      transpose_cache_(1024),
      topology_(GetTopology(platform_name_, std::move(gpu_topology),
                            addressable_devices_)),
      kv_store_(std::move(kv_store)),
      owned_devices_(std::move(devices)),
      compile_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), tsl::ThreadOptions(),
          "TfrtGpuClient_compile_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()),
          true)),
      blocking_thread_pool_(std::make_unique<UnboundedAsyncWorkRunner>(
          "TfrtGpuClient_blocking_thread_pool")),
      non_blocking_thread_pool_(std::make_unique<UnboundedAsyncWorkRunner>(
          "TfrtGpuClient_non_blocking_thread_pool")) {
  LOG(INFO) << "TfrtGpuClient created with " << addressable_devices_.size()
            << " / " << devices_.size() << " addressable devices.";
}

TfrtGpuClient::~TfrtGpuClient() {
  // Destroy objects that may invoke CUDA APIs (e.g., allocators) on a separate
  // thread pool. See the comments on `TfrtGpuThreadChecker` for more info.
  absl::WrapUnique(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "TfrtGpuClientDestructor", [&]() {
        // Thread pools must be destructed first, to make all the pending tasks
        // are completed before the client is destructed.
        compile_thread_pool_ = {};
        blocking_thread_pool_ = {};
        non_blocking_thread_pool_ = {};

        // Destructed after the thread pools, to ensure that all kernels in the
        // streams are finished.
        owned_devices_.clear();
        owned_memory_spaces_.clear();

        host_memory_allocator_ = {};
        allocator_ = {};
      }));

  LOG(INFO) << "TfrtGpuClient destroyed.";
}

absl::string_view TfrtGpuClient::platform_version() const {
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#if TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)  // rocm
  // TF_ROCM_VERSION format may change in future. Use it
  // cautiously
  return "rocm " STRINGIFY(TF_ROCM_VERSION);
#elif GOOGLE_CUDA && defined(CUDART_VERSION)  // cuda
  return "cuda " STRINGIFY(CUDART_VERSION);
#else
  return "<unknown>";
#endif  // TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)
}

absl::StatusOr<PjRtDevice*> TfrtGpuClient::LookupDevice(
    PjRtGlobalDeviceId global_device_id) const {
  auto it = id_to_device_.find(global_device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         global_device_id.value());
}

absl::StatusOr<PjRtDevice*> TfrtGpuClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  for (auto* device : addressable_devices_) {
    if (local_device_id == device->local_device_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_hardware_id %d",
                         local_device_id.value());
}

void TfrtGpuClient::UpdateGlobalProcessInfo(
    absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) {
  if (!abort_collectives_on_failure_) {
    // If we're not aborting collectives, we don't need to track information
    // about other processes. We only track global process info to know when to
    // abort.
    VLOG(5) << "Not updating global process info because "
               "abort_collectives_on_failure_ is false";
    return;
  }
  absl::Status s = ::xla::gpu::UpdateGlobalProcessInfo(infos);
  if (!s.ok()) {
    LOG(WARNING) << "Failed to update global process info: " << s;
  }
}

absl::StatusOr<Layout> TfrtGpuClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  return topology_.GetDefaultLayout(element_type, dims);
}

absl::StatusOr<std::unique_ptr<HloCostAnalysis>>
TfrtGpuClient::GetHloCostAnalysis() const {
  return std::make_unique<HloCostAnalysis>(
      xla_client_->backend().compiler()->ShapeSizeBytesFunction());
}

absl::Span<PjRtMemorySpace* const> TfrtGpuClient::memory_spaces() const {
  return memory_spaces_;
}

std::optional<PjRtPluginAttributes> TfrtGpuClient::plugin_attributes() const {
  PjRtPluginAttributes attributes =
      PjRtClient::plugin_attributes().value_or(PjRtPluginAttributes());
  attributes.pjrt_c_api_major_version = 0;
  attributes.pjrt_c_api_minor_version = 0;
  attributes.attributes["serialize_with_sdy"] = PjRtValueType(true);
  attributes.attributes["supports_cross_host_transfers"] = PjRtValueType(true);
  return attributes;
}

absl::StatusOr<DeviceAssignment> TfrtGpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  return Compile(computation, options, /*lookup_addressable_devices=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    const XlaComputation& computation, CompileOptions options,
    bool lookup_addressable_devices) {
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
                         options, lookup_addressable_devices);

  // TODO: b/382117736 - Record free gpu memory.
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/gpu/se_gpu_pjrt_client.cc#L792-L809
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options, bool lookup_addressable_devices) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::CompileInternal");
  VLOG(1) << "TfrtGpuClient::CompileInternal";
  if (key_value_store().has_value() &&
      !options.executable_build_options.key_value_store()) {
    options.executable_build_options.set_key_value_store(*key_value_store());
  }
  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  TF_RETURN_IF_ERROR(
      UpdateCompileOptions(&options, lookup_addressable_devices));

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

  return BuildPjRtExecutable(computation.proto(), std::move(local_executables),
                             std::move(input_options));
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  TF_RETURN_IF_ERROR(pjrt::MaybeDumpCompileInputs(options, module, topology_));
  return Compile(module, options, /*lookup_addressable_devices=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options,
    bool lookup_addressable_devices) {
  XlaComputation xla_computation;
  ExecutableBuildOptions& exec_build_options = options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, &exec_build_options,
      mlir::mhlo::getGpuChloToHighLevelMhloOptions()));

  // If the compile options specify argument layout, then let's
  // fall back to using the options to determine layouts.
  if (options.argument_layouts) {
    return Compile(xla_computation, options, lookup_addressable_devices);
  }

  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                      GetArgLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                      GetOutputLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                      GetArgMemoryKinds(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                      GetOutputMemoryKinds(module));

  // If auto-sharding modifies shapes of arguments and/or result,
  // we get a callback to restore the layouts. Let us restore the layouts
  // according to the attributes we parsed from MLIR.
  auto layout_callback = [local_client = xla_client_, &arg_layout_modes,
                          &out_layout_modes, &arg_memory_spaces,
                          &out_memory_spaces](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    XlaComputation xla_computation(XlaComputation(module.ToProto()));
    return LayoutModesToXlaShapes(
        xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
        out_memory_spaces,
        [local_client](Shape shape) -> absl::StatusOr<Shape> {
          return local_client->backend()
              .transfer_manager()
              ->ChooseCompactLayoutForShape(shape);
        });
  };

  // This call will update result_layout in options.executable_build_options.
  TF_ASSIGN_OR_RETURN(auto arg_layouts_and_pointers,
                      LayoutModesToXla(
                          xla_computation, arg_layout_modes, out_layout_modes,
                          arg_memory_spaces, out_memory_spaces,
                          [this](Shape shape) -> absl::StatusOr<Shape> {
                            return this->xla_client_->backend()
                                .transfer_manager()
                                ->ChooseCompactLayoutForShape(shape);
                          },
                          options.executable_build_options));
  return CompileInternal(xla_computation, arg_layouts_and_pointers.second,
                         layout_callback, options, lookup_addressable_devices);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(const XlaComputation& computation,
                              CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      Compile(computation, options,
                              /*lookup_addressable_devices=*/true));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      Compile(module, options,
                              /*lookup_addressable_devices=*/true));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  CHECK_EQ(memory_space->devices().size(), 1);
  auto* device = memory_space->devices().front();
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  se::DeviceAddressBase device_memory(device_ptr, byte_size);
  auto non_owning_buffer = GpuDeviceMemory(device_memory);
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<GpuDeviceMemory>(
          std::move(non_owning_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      /*definition_event=*/tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      /*ready_event=*/tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      std::move(on_delete_callback));
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateUninitializedBuffer(const Shape& shape,
                                         PjRtMemorySpace* memory_space) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::CreateUninitializedBuffer");
  VLOG(1) << "TfrtGpuClient::CreateUninitializedBuffer: shape: "
          << shape.ToString()
          << " memory_space: " << memory_space->DebugString();
  TransferManager* transfer_manager =
      xla_client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));
  return AllocateTfrtGpuDestinationBuffer(
      compact_shape, tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      absl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]), this,
      memory_space);
}

absl::StatusOr<
    std::pair<std::unique_ptr<PjRtBuffer>, PjRtFulfillAliasBufferCallback>>
TfrtGpuClient::CreateAliasBuffer(const Shape& shape,
                                 PjRtMemorySpace* memory_space) {
  auto buffer_promise = tsl::MakeIndirectAsyncValue();
  auto definition_event_promise = tsl::MakeIndirectAsyncValue();
  auto ready_event_promise = tsl::MakeIndirectAsyncValue();

  auto tracked_device_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
      tsl::AsyncValueRef<GpuDeviceMemory>(buffer_promise),
      tsl::AsyncValueRef<GpuEvent>(definition_event_promise),
      tsl::AsyncValueRef<GpuEvent>(ready_event_promise));

  if (memory_space->devices().size() != 1) {
    return absl::InternalError(
        "CreateAliasBuffer only supports single-device memory spaces");
  }
  auto* device = tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]);
  auto result_buffer = std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this, device, memory_space);

  xla::PjRtFulfillAliasBufferCallback fulfill_alias_buffer_cb =
      [buffer_promise = std::move(buffer_promise),
       definition_event_promise = std::move(definition_event_promise),
       ready_event_promise = std::move(ready_event_promise)](
          absl::StatusOr<xla::PjRtBuffer*> buffer_or) -> absl::Status {
    if (!buffer_or.ok()) {
      definition_event_promise->SetError(buffer_or.status());
      ready_event_promise->SetError(buffer_or.status());
      buffer_promise->SetError(buffer_or.status());
      return buffer_or.status();
    }
    auto* tfrt_buffer = tsl::down_cast<xla::TfrtGpuBuffer*>(*buffer_or);
    if (tfrt_buffer == nullptr) {
      auto status = absl::InternalError("Failed to cast to TfrtGpuBuffer");
      definition_event_promise->SetError(status);
      ready_event_promise->SetError(status);
      buffer_promise->SetError(status);
      return status;
    }
    {
      absl::MutexLock lock(tfrt_buffer->mu_);
      xla::TrackedGpuDeviceBuffer* tracked_gpu_buffer =
          tfrt_buffer->tracked_device_buffer_.get();
      buffer_promise->ForwardTo(tracked_gpu_buffer->buffer().CopyRCRef());
      definition_event_promise->ForwardTo(
          tracked_gpu_buffer->definition_event().CopyRCRef());
      ready_event_promise->ForwardTo(
          tracked_gpu_buffer->ready_event().CopyRCRef());
    }

    return absl::OkStatus();
  };

  return std::make_pair(std::move(result_buffer),
                        std::move(fulfill_alias_buffer_cb));
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
TfrtGpuClient::BuildPjRtExecutable(
    std::optional<HloModuleProto> unoptimized_hlo_module_proto,
    std::vector<std::unique_ptr<LocalExecutable>> local_executables,
    CompileOptions compile_options) {
  if (local_executables.empty()) {
    return Internal("No local executable");
  }
  if (local_executables.size() != 1) {
    return Unimplemented("Multiple executables are not supported");
  }
  Executable* built_executable = local_executables[0]->executable();
  if (!built_executable->has_module()) {
    return absl::InternalError("Executable does not have HLO modules.");
  }
  const auto& hlo_module = built_executable->module();

  const int num_replicas = hlo_module.config().replica_count();
  const int num_partitions = hlo_module.config().num_partitions();
  const std::string name = hlo_module.name();
  const std::string fingerprint = hlo_module.GetFingerprint128();

  const auto& result_shape =
      local_executables[0]->executable()->module().result_shape();
  if (result_shape.IsTuple()) {
    for (auto& leaf_shape : result_shape.tuple_shapes()) {
      if (leaf_shape.IsTuple()) {
        return absl::InternalError(
            absl::StrCat("Nested tuples are not supported with "
                         "TfrtGpuClient. got: ",
                         result_shape.ToString()));
      }
    }
  }

  return std::make_unique<StreamExecutorExecutable>(
      std::move(compile_options), std::move(unoptimized_hlo_module_proto),
      std::move(local_executables), xla_client_, num_replicas, num_partitions,
      name, fingerprint, memory_spaces()[0]->kind());
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
TfrtGpuClient::DeserializeExecutable(
    absl::string_view serialized,
    std::optional<CompileOptions> compile_options) {
  TF_ASSIGN_OR_RETURN(
      auto local_executables_and_options,
      DeserializeToLocalExecutable(serialized, compile_options));

  return BuildPjRtExecutable(/*unoptimized_hlo_module_proto=*/std::nullopt,
                             std::move(local_executables_and_options.first),
                             local_executables_and_options.second);
}

namespace {
absl::StatusOr<ExecutableAndOptionsProto> DeserializeExecutableAndOptionsProto(
    absl::string_view serialized) {
  ExecutableAndOptionsProto proto;
  auto reader = std::make_unique<riegeli::StringReader<>>(serialized);
  // The serialized string may be of the new SplitProto format (which allows
  // executables larger than 2GB) or the legacy format which is just a regular
  // proto.
  ASSIGN_OR_RETURN(bool is_split_proto, IsSplitProto(*reader));
  if (is_split_proto) {
    RETURN_IF_ERROR(ReadSplitProto(std::move(reader), proto));
    return proto;
  }

  if (serialized.size() > std::numeric_limits<int>::max()) {
    return Internal("Proto is too large (>2GB)");
  }
  if (!proto.ParseFromString(serialized)) {
    return Internal("Proto deserialization failed");
  }

  return proto;
}
}  // namespace

absl::StatusOr<
    std::pair<std::vector<std::unique_ptr<LocalExecutable>>, CompileOptions>>
TfrtGpuClient::DeserializeToLocalExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options) {
  TF_ASSIGN_OR_RETURN(ExecutableAndOptionsProto proto,
                      DeserializeExecutableAndOptionsProto(serialized));

  if (!proto.pjrt_client_name().empty() &&
      proto.pjrt_client_name() != kPjRtClientName) {
    return Internal(
        "Serialized executable is from an incompatible PjRt client type. "
        "PjRt client type expected by the serialized executable: %s",
        proto.pjrt_client_name());
  }

  CompileOptions compile_options;
  if (options.has_value()) {
    compile_options = *std::move(options);
  } else {
    TF_ASSIGN_OR_RETURN(compile_options,
                        CompileOptions::FromProto(proto.compile_options()));
  }

  tsl::profiler::TraceMe traceme("TfrtGpuClient::DeserializeToLocalExecutable");
  VLOG(1) << "TfrtGpuClient::DeserializeToLocalExecutable";

  std::string str = std::move(*proto.mutable_serialized_executable());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalExecutable> loaded,
      xla_client_->Load(str, compile_options.executable_build_options));

  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.push_back(std::move(loaded));

  return std::make_pair(std::move(local_executables), compile_options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::LoadSerializedExecutable(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
  TF_ASSIGN_OR_RETURN(auto local_executables_and_options,
                      DeserializeToLocalExecutable(serialized, options));
  return LoadInternal(std::move(local_executables_and_options.first),
                      local_executables_and_options.second);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::LoadInternal(
    std::vector<std::unique_ptr<LocalExecutable>> local_executables,
    CompileOptions compile_options) {
  auto input_options = compile_options;

  TF_RETURN_IF_ERROR(compile_options.ApplyAllOptionOverrides());

  TF_ASSIGN_OR_RETURN(
      ExecutableExtras extras,
      UpdateCompileOptionsAndGetExecutableExtras(&compile_options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  const auto& ex_options = compile_options.executable_build_options;
  const bool xla_dump_hlo_unoptimized_snapshots =
      ex_options.has_debug_options() &&
      ex_options.debug_options().xla_dump_hlo_unoptimized_snapshots();
  HloModuleProto hlo_module_proto;
  if (xla_dump_hlo_unoptimized_snapshots) {
    hlo_module_proto = local_executables[0]->executable()->module().ToProto();
  }

  auto executable = std::make_unique<TfrtGpuExecutable>(
      std::move(local_executables),
      compile_options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(compile_options.parameter_is_tupled_arguments));
  if (xla_dump_hlo_unoptimized_snapshots) {
    executable->SetInputHloSnapshotBits(
        std::move(hlo_module_proto),
        compile_options.executable_build_options.debug_options());
  }
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtGpuClient::Load(
    std::unique_ptr<PjRtExecutable> executable,
    const LoadOptions& load_options) {
  auto se_executable = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorExecutable*>(executable.release()));
  CompileOptions compile_options = se_executable->compile_options();

  tsl::profiler::TraceMe traceme("TfrtGpuClient::Load");
  VLOG(1) << "TfrtGpuClient::Load";

  TF_ASSIGN_OR_RETURN(auto local_executable, se_executable->ConsumeExecutable(
                                                 xla_client_, compile_options));
  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.push_back(std::move(local_executable));
  return LoadInternal(std::move(local_executables), compile_options);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::CreateErrorBuffer(
    absl::Status error, const Shape& shape, PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  if (memory_space->client() != this) {
    return absl::InvalidArgumentError(
        "Memory space is not attached to this client");
  }

  if (IsMemorySpaceKind<UnpinnedHostMemorySpace>(memory_space)) {
    return absl::InvalidArgumentError(
        "Error buffers are not supported for unpinned host memory yet");
  }

  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(memory_space->devices().front());
  VLOG(4) << "TfrtGpuClient::CreateErrorBuffer: shape: " << shape.ToString()
          << " device: " << device->DebugString() << " error: " << error;

  auto error_async_value_ref = tsl::MakeErrorAsyncValueRef(error);
  auto tracked_device_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
      /*buffer=*/error_async_value_ref,
      /*definition_event=*/error_async_value_ref,
      /*ready_event=*/error_async_value_ref);
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::Status TfrtGpuClient::UpdateCompileOptions(
    CompileOptions* options, bool lookup_addressable_devices) {
  return UpdateCompileOptionsInternal(options, /*returned_extras=*/nullptr,
                                      lookup_addressable_devices);
}

absl::StatusOr<TfrtGpuClient::ExecutableExtras>
TfrtGpuClient::UpdateCompileOptionsAndGetExecutableExtras(
    CompileOptions* options) {
  ExecutableExtras extras;
  TF_RETURN_IF_ERROR(UpdateCompileOptionsInternal(
      options, &extras, /*lookup_addressable_devices=*/true));
  return extras;
}

absl::Status TfrtGpuClient::UpdateCompileOptionsInternal(
    CompileOptions* options, ExecutableExtras* returned_extras,
    bool lookup_addressable_devices) {
  ExecutableBuildOptions& build_options = options->executable_build_options;
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(compile_thread_pool_.get());
  }
  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(allocator());
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

  // We don't look up devices when it is not required. It could fail if
  // we look up a device ID on a client with a different topology.
  // Note that we always look up devices for XLA GPU shard autotuning, as it
  // needs to know the number of processes and the current process index.
  const bool use_xla_gpu_shard_autotuning =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_gpu_shard_autotuning();
  if (!lookup_addressable_devices && !use_xla_gpu_shard_autotuning) {
    if (build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(0);
    }
    return absl::OkStatus();
  }

  ExecutableExtras extras;
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

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
          VLOG(4) << "Non-local device: " << device_id;
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
      if (build_options.device_ordinal() < 0) {
        build_options.set_device_ordinal(0);
      }
    } else {
      if (build_options.device_ordinal() < 0) {
        build_options.set_device_ordinal(
            addressable_devices.front()->local_hardware_id().value());
      }
      build_options.set_process_index(*this_process_index);
      build_options.set_process_count(all_process_indices.size());
    }
  }
  if (returned_extras != nullptr) {
    *returned_extras = std::move(extras);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  TfrtGpuDevice* device =
      tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]);

  tsl::profiler::TraceMe traceme("TfrtGpuClient::BufferFromHostBuffer");
  Shape device_shape = ShapeUtil::MakeShape(type, dims);

  VLOG(3) << "TfrtGpuClient::BufferFromHostBuffer: shape: "
          << device_shape.ToString() << " device: " << device->DebugString();

  absl::InlinedVector<int64_t, 4> tmp_strides;
  if (!byte_strides) {
    tmp_strides.resize(dims.size());
    TF_RETURN_IF_ERROR(ShapeUtil::UnpackedByteStrides(
        device_shape, absl::MakeSpan(tmp_strides)));
    byte_strides = tmp_strides;
  }

  int64_t byte_size = ShapeUtil::ByteSizeOf(device_shape);

  TransferManager* transfer_manager = xla_client_->backend().transfer_manager();
  if (device_layout != nullptr) {
    *(device_shape.mutable_layout()) = *device_layout;
  } else {
    TF_ASSIGN_OR_RETURN(
        device_shape,
        transfer_manager->ChooseCompactLayoutForShape(device_shape));
  }

  absl::InlinedVector<int64_t, 4> shape_strides(
      device_shape.dimensions().size());
  TF_RETURN_IF_ERROR(ShapeUtil::UnpackedByteStrides(
      device_shape, absl::MakeSpan(shape_strides)));
  bool host_and_device_strides_equal =
      (byte_size == 0 || *byte_strides == shape_strides);

  std::shared_ptr<TransposePlan> transpose;
  if (!host_and_device_strides_equal) {
    absl::InlinedVector<int64_t, 4> permutation(dims.size());
    absl::c_reverse_copy(device_shape.layout().minor_to_major(),
                         permutation.begin());
    TransposePlan::Options options;
    options.elem_size_in_bytes = primitive_util::ByteWidth(type);
    options.dims = dims;
    options.permutation = permutation;
    options.input_striding = TransposePlan::Striding{*byte_strides};
    absl::MutexLock lock(transpose_mu_);
    TF_ASSIGN_OR_RETURN(transpose, transpose_cache_.GetOrCreate(options));
  }

  bool should_pack = primitive_util::IsSubByteNonPredType(type) &&
                     transfer_manager->PackSubbyteTypes();
  int64_t packed_size;
  if (should_pack) {
    packed_size =
        CeilOfRatio<int64_t>(byte_size, 8 / primitive_util::BitWidth(type));
  } else {
    packed_size = byte_size;
  }
  auto dst_definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TfrtGpuBuffer> output_buffer,
                      AllocateTfrtGpuDestinationBuffer(
                          device_shape, dst_definition_event.CopyRef(), device,
                          this, memory_space, packed_size));
  auto copy_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TrackedGpuDeviceBuffer* allocated_dst_buffer =
      output_buffer->AcquireUsage(copy_event);
  CHECK(allocated_dst_buffer != nullptr);
  auto gpu_buffer = allocated_dst_buffer->buffer();
  if (gpu_buffer.IsError()) {
    return gpu_buffer.GetError();
  }

  // If necessary, allocate a host-side buffer for staging host-to-device
  // transfers. On GPU this is a buffer in pinned memory.
  HostMemoryAllocator::OwnedPtr staging_buffer;
  bool must_use_staging_buffer =
      host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall ||
      !host_and_device_strides_equal || packed_size != byte_size;
  bool use_staging_buffer = must_use_staging_buffer ||
                            ShouldStageHostToDeviceTransfers(data, packed_size);

  auto copy_to_staging_buffer = [allocator = host_memory_allocator(), byte_size,
                                 type, packed_size,
                                 transpose{std::move(transpose)},
                                 should_pack](const void* src_buf) mutable {
    tsl::profiler::TraceMe traceme("BufferFromHostBuffer::H2D_staging_copy");

    HostMemoryAllocator::OwnedPtr staging_buffer =
        allocator->Allocate(transpose ? byte_size : packed_size);
    void* buffer = staging_buffer.get();
    const void* data_ptr = src_buf;
    VLOG(3) << "H2D staging copy: " << src_buf << " -> " << buffer << "("
            << byte_size << " -> " << packed_size << " bytes)";

    if (transpose) {
      transpose->Execute(data_ptr, buffer);
      data_ptr = buffer;
    }
    if (should_pack) {
      primitive_util::PackIntN(
          type,
          absl::MakeConstSpan(static_cast<const char*>(data_ptr), byte_size),
          absl::MakeSpan(static_cast<char*>(buffer), packed_size));
      data_ptr = buffer;
    }
    if (data_ptr != buffer) {
      std::memcpy(buffer, data_ptr, byte_size);
    }
    VLOG(3) << "H2D staging copy done";
    return staging_buffer;
  };

  auto h2d_do_copy = [device, packed_size, copy_event(std::move(copy_event)),
                      dst_definition_event(std::move(dst_definition_event)),
                      gpu_buffer{gpu_buffer.CopyRef()}](const void* src_buf) {
    tsl::profiler::TraceMe traceme([&] {
      return tsl::profiler::TraceMeEncode(
          "BufferFromHostBuffer::H2D_GPU_copy",
          {{"device", device->id()}, {"size", packed_size}});
    });
    auto stream = device->stream();

    se::DeviceAddressBase dest = gpu_buffer->buffer();
    VLOG(3) << "H2D copy: " << src_buf << " -> " << dest.opaque() << " ("
            << packed_size << " bytes) on device " << device->DebugString();

    absl::Status status = stream->Memcpy(&dest, src_buf, packed_size);

    if (!status.ok()) {
      copy_event.SetError(status);
      dst_definition_event.SetError(status);
      return;
    }

    status = BlockHostUntilDoneWithHostCallback(stream);
    VLOG(3) << "H2D copy done. " << status;

    if (status.ok()) {
      copy_event.SetStateConcrete();
      dst_definition_event.SetStateConcrete();
    } else {
      copy_event.SetError(status);
      dst_definition_event.SetError(status);
    }
  };

  // Define H2D copy lambda. First, copy host data to staging buffer, then copy
  // staging buffer to GPU device.
  auto h2d_copy = [this, use_staging_buffer, data,
                   on_done_with_host_buffer =
                       std::move(on_done_with_host_buffer),
                   copy_to_staging_buffer(std::move(copy_to_staging_buffer)),
                   h2d_do_copy(std::move(h2d_do_copy))]() mutable {
    if (use_staging_buffer) {
      // Copy to the target data to staging buffer first.
      HostMemoryAllocator::OwnedPtr staging_buffer;
      staging_buffer = copy_to_staging_buffer(data);

      // Call on_done_with_host_buffer to release the data buffer.
      if (on_done_with_host_buffer) {
        std::move(on_done_with_host_buffer)();
      }

      // Copy the data from the staging buffer to GPU.
      blocking_thread_pool_->Schedule(
          [h2d_do_copy(std::move(h2d_do_copy)),
           staging_buffer(std::move(staging_buffer))]() {
            h2d_do_copy(staging_buffer.get());
          });
    } else {
      blocking_thread_pool_->Schedule(
          [h2d_do_copy(std::move(h2d_do_copy)), data,
           on_done_with_host_buffer =
               std::move(on_done_with_host_buffer)]() mutable {
            // Copy the data directly to GPU.
            h2d_do_copy(data);

            // Call on_done_with_host_buffer to release the data buffer.
            if (on_done_with_host_buffer) {
              std::move(on_done_with_host_buffer)();
            }
          });
    }
  };

  if (host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall) {
    h2d_copy();
  } else {
    non_blocking_thread_pool_->Schedule(std::move(h2d_copy));
  }

  return output_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  VLOG(4) << "TfrtGpuClient::CreateBuffersForAsyncHostToDevice";
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices()[0];
  auto* tfrt_gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
  return TfrtGpuAsyncHostToDeviceTransferManager::Create(
      shape_specs, device_layouts, tfrt_gpu_device, this, memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                     PjRtMemorySpace* memory_space,
                                     const Layout* device_layout) {
  if (device_layout) {
    return absl::UnimplementedError(absl::StrCat(
        "BufferFromHostLiteral with device_layout is not implemented on "
        "platform: ",
        platform_name()));
  }
  PjRtDevice* device = memory_space->devices()[0];
  tsl::profiler::TraceMe traceme("TfrtGpuClient::BufferFromHostLiteral");

  VLOG(3) << "TfrtGpuClient::BufferFromHostLiteral: shape: "
          << literal.shape().ToString() << " device: " << device->DebugString();

  const Shape& shape = literal.shape();
  if (shape.IsTuple()) {
    return Unimplemented(
        "Tuple case is not supported in TfrtGpuClient::BufferFromHostLiteral");
  }

  // Add a placeholder definition event for each leaf buffer when creating the
  // buffer. They are set only after h2d dispatch.
  tsl::AsyncValueRef<GpuEvent> definition_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfrtGpuBuffer> output_buffer,
      AllocateTfrtGpuDestinationBuffer(shape, definition_event,
                                       tsl::down_cast<TfrtGpuDevice*>(device),
                                       this, memory_space));

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = output_buffer->AcquireUsage(usage_event);
  CHECK(device_buffer);

  // It is OK to capture `buffer` pointer because the `output_buffer` can't
  // be deleted until all the usage holds have gone away.
  VLOG(4) << "BufferFromHostLiteral for device_buffer: " << device_buffer;
  non_blocking_thread_pool_->Schedule(
      [literal, definition_event, device_buffer, shape, this,
       device = tsl::down_cast<TfrtGpuDevice*>(device),
       usage_event = std::move(usage_event)]() mutable {
        tsl::profiler::TraceMe traceme("BufferFromHostLiteral::H2D_Dispatch");
        TransferManager* transfer_manager =
            xla_client()->backend().transfer_manager();

        auto stream = device->stream();

        PrimitiveType type = literal.shape().element_type();
        bool should_pack = primitive_util::IsSubByteNonPredType(type) &&
                           transfer_manager->PackSubbyteTypes();
        int64_t byte_size = literal.size_bytes();
        if (should_pack) {
          byte_size = CeilOfRatio<int64_t>(byte_size,
                                           8 / primitive_util::BitWidth(type));
        }
        const auto& buffer = device_buffer->buffer();
        if (literal.shape().IsArray()) {
          CHECK_EQ(byte_size, buffer->size_bytes());
        }

        ShapedBuffer shaped_buffer = buffer->AsShapedBuffer(shape, device);

        CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(stream, literal,
                                                                shaped_buffer));

        absl::Status status = BlockHostUntilDoneWithHostCallback(stream);
        CHECK_OK(status) << "Failed to block host until done";
        VLOG(3) << "BufferFromHostLiteral done for device_buffer: "
                << device_buffer;

        definition_event.SetStateConcrete();
        usage_event.SetStateConcrete();
      });
  return std::unique_ptr<PjRtBuffer>(std::move(output_buffer));
}

absl::Status TfrtGpuClient::DmaMap(void* data, size_t buffer_size) {
  return RunOnAsyncWorkRunner(
      non_blocking_thread_pool(), [&]() -> absl::Status {
        tsl::profiler::TraceMe trace_me("TfrtGpuClient::DmaMap");
        se::StreamExecutor* executor =
            tensorflow::down_cast<TfrtGpuDevice*>(addressable_devices_[0])
                ->executor();
        DCHECK(executor);
        bool success = executor->HostMemoryRegister(data, buffer_size);
        if (!success) {
          return absl::InternalError(absl::StrFormat(
              "Failed to register host memory at address: %ps", data));
        }
        absl::MutexLock lock(dma_maps_mutex_);
        dma_maps_.insert({data, buffer_size});
        return absl::OkStatus();
      });
}

absl::Status TfrtGpuClient::DmaUnmap(void* data) {
  return RunOnAsyncWorkRunner(
      non_blocking_thread_pool(), [&]() -> absl::Status {
        tsl::profiler::TraceMe trace_me("TfrtGpuClient::DmaUnmap");
        se::StreamExecutor* executor =
            tensorflow::down_cast<TfrtGpuDevice*>(addressable_devices_[0])
                ->executor();
        DCHECK(executor);
        bool success = executor->HostMemoryUnregister(data);
        if (!success) {
          return absl::InternalError(absl::StrFormat(
              "Failed to unregister host memory at address: %ps", data));
        }
        absl::MutexLock lock(dma_maps_mutex_);
        dma_maps_.erase(data);
        return absl::OkStatus();
      });
}

bool TfrtGpuClient::IsDmaMapped(const void* data_start, int64_t transfer_size) {
  absl::MutexLock lock(dma_maps_mutex_);
  if (dma_maps_.empty()) {
    return false;
  }
  auto it = dma_maps_.lower_bound(data_start);
  if (it == dma_maps_.end()) {
    return false;
  }
  void* data_end = (char*)data_start + transfer_size;
  void* map_end = (char*)it->first + it->second;
  return data_end <= map_end;
}

namespace {

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClientInternal(
    const GpuClientOptions& options) {
#if TENSORFLOW_USE_ROCM
  const auto* pjrt_platform_name = xla::RocmName();
#elif TENSORFLOW_USE_SYCL
  const auto* pjrt_platform_name = xla::SyclName();
#else   // TENSORFLOW_USE_ROCM
  const auto* pjrt_platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM

  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  EnablePeerAccess(xla_client->backend().stream_executors());

  HostMemoryAllocator::Factory host_memory_allocator_factory =
      options.host_memory_allocator_factory;
  if (host_memory_allocator_factory == nullptr &&
      !xla_client->backend().stream_executors().empty()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<tsl::Allocator> allocator,
        GetGpuHostAllocator(xla_client->backend().stream_executors().front()));
    host_memory_allocator_factory =
        [allocator = allocator.release()](
            const HostMemoryAllocator::Options& options) mutable {
          return std::make_unique<BasicHostMemoryAllocator>(
              absl::WrapUnique(allocator), tsl::Allocator::kAllocatorAlignment);
        };
  }

  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  if (options.enable_mock_nccl) {
    gpu_run_options->set_enable_mock_collectives();
  }

  static const bool xla_gpu_require_exclusive_lock =
      xla::GetDebugOptionsFromFlags().xla_gpu_require_exclusive_lock();
  if (xla_gpu_require_exclusive_lock) {
    gpu_run_options->set_requires_exclusive_lock_on_gpu();
  }

  std::shared_ptr<KeyValueStoreInterface> kv_store = options.kv_store;
  if (options.enable_mock_nccl) {
    kv_store = std::make_shared<InMemoryKeyValueStore>();
  }
  TF_RET_CHECK(options.num_nodes == 1 || kv_store != nullptr);
  TF_ASSIGN_OR_RETURN(
      DeviceTopologyPair device_topology_pair,
      BuildDistributedDevices(
          pjrt_platform_name, xla_client, options.node_id,
          options.max_inflight_computations, options.num_nodes,
          gpu_run_options.get(), kv_store, options.enable_mock_nccl,
          options.mock_gpu_topology, options.partition_index, absl::Minutes(2),
          absl::Minutes(5)));

  std::vector<std::unique_ptr<TfrtGpuDevice>> devices =
      std::move(device_topology_pair.first);
  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(device_topology_pair.second));

  TF_ASSIGN_OR_RETURN(
      auto allocator,
      CreateDeviceAllocator(xla_client, options.allocator_config, devices));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtGpuClient>(
      std::move(pjrt_platform_name), options.node_id, xla_client,
      std::move(devices), options.should_stage_host_to_device_transfers,
      options.abort_collectives_on_failure, std::move(allocator),
      std::move(host_memory_allocator_factory), std::move(gpu_run_options),
      std::move(kv_store), std::move(gpu_topology)));
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options) {
  absl::StatusOr<std::unique_ptr<PjRtClient>> result;
  {
    // Bounce through a thread to avoid calling CUDA inline.
    absl::WrapUnique(tsl::Env::Default()->StartThread(
        {}, "GetTfrtGpuClient",
        [&]() { result = GetTfrtGpuClientInternal(options); }));
  }
  return result;
}

}  // namespace xla
