/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/python/py_client.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/callback.h"
#include "xla/python/guard_lib.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pprof_profile_builder.h"
#include "xla/python/py_array.h"
#include "xla/python/py_device.h"
#include "xla/python/py_executable.h"
#include "xla/python/py_host_callback.h"
#include "xla/python/py_memory_space.h"
#include "xla/python/py_values.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/traceback.h"
#include "xla/python/types.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/platform_util.h"  // IWYU pragma: keep
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace nb = nanobind;

/*static*/ nb_class_ptr<PyClient> PyClient::Make(
    std::shared_ptr<ifrt::Client> ifrt_client) {
  auto client = make_nb_class<PyClient>(std::move(ifrt_client));
  Initialize(client);
  return client;
}

PyClient::PyClient(std::shared_ptr<ifrt::Client> ifrt_client)
    : ifrt_client_(std::move(ifrt_client)),
      client_attributes_(ifrt_client_->Attributes()) {
  CHECK(ifrt_client_);
}

/* static */ void PyClient::Initialize(nb_class_ptr<PyClient> client) {
  for (ifrt::Device* device : client->ifrt_client()->devices()) {
    client->devices_[device] = make_nb_class<PyDevice>(client, device);

    for (ifrt::Memory* memory : device->Memories()) {
      auto& py_memory = client->memory_spaces_[memory];
      if (py_memory.get() == nullptr) {
        py_memory = make_nb_class<PyMemorySpace>(client, memory);
      }
    }
  }
}

PyClient::~PyClient() {
  nb::gil_scoped_release gil;
  ifrt_client_ = nullptr;
}

nb_class_ptr<PyDevice> PyClient::GetPyDevice(ifrt::Device* device) {
  auto& py_device = devices_[device];
  if (py_device.get() == nullptr) {
    py_device = make_nb_class<PyDevice>(
        nb::borrow<nb_class_ptr<PyClient>>(nb::find(this)), device);
  }
  return py_device;
}

nb_class_ptr<PyMemorySpace> PyClient::GetPyMemorySpace(
    ifrt::Memory* memory_space) {
  auto& py_memory = memory_spaces_[memory_space];
  if (py_memory.get() == nullptr) {
    py_memory = make_nb_class<PyMemorySpace>(
        nb::borrow<nb_class_ptr<PyClient>>(nb::find(this)), memory_space);
  }
  return py_memory;
}

std::vector<nb_class_ptr<PyDevice>> PyClient::Devices() {
  std::vector<nb_class_ptr<PyDevice>> devices;
  auto span = ifrt_client_->devices();
  devices.reserve(span.size());
  for (ifrt::Device* device : span) {
    devices.push_back(GetPyDevice(device));
  }
  return devices;
}

std::vector<nb_class_ptr<PyDevice>> PyClient::LocalDevices() {
  std::vector<nb_class_ptr<PyDevice>> devices;
  devices.reserve(ifrt_client_->addressable_devices().size());
  for (ifrt::Device* device : ifrt_client_->addressable_devices()) {
    devices.push_back(GetPyDevice(device));
  }
  return devices;
}

std::vector<nb_class_ptr<PyDevice>> PyClient::GetAllDevices() {
  std::vector<nb_class_ptr<PyDevice>> devices;
  devices.reserve(ifrt_client_->GetAllDevices().size());
  for (ifrt::Device* device : ifrt_client_->GetAllDevices()) {
    devices.push_back(GetPyDevice(device));
  }
  return devices;
}

absl::StatusOr<nb_class_ptr<PyDevice>> PyClient::DeviceFromLocalHardwareId(
    int local_hardware_id) {
  TF_ASSIGN_OR_RETURN(ifrt::Device * device,
                      ifrt_client_->LookupAddressableDevice(local_hardware_id));
  return GetPyDevice(device);
}

nb::list PyClient::LiveExecutables() {
  CHECK(PyGILState_Check());
  nb::ft_lock_guard lock(executables_mutex_);
  nb::list executables;
  for (PyLoadedExecutable* exec = executables_; exec; exec = exec->next_) {
    if (!exec->is_deleted()) {
      executables.append(nb::find(exec));
    }
  }
  return executables;
}

absl::Status PyClient::Defragment() {
  CHECK(PyGILState_Check());
  if (!llvm::isa<ifrt::PjRtCompatibleClient>(ifrt_client_.get())) {
    return absl::UnimplementedError(
        "Defragmentation is not supported on this runtime.");
  }
  ifrt::PlatformId platform_id = ifrt_client_->platform_id();
  bool is_gpu_client = platform_id == CudaId() || platform_id == RocmId() ||
                       platform_id == SyclId();

  if (!is_gpu_client) {
    return pjrt_client()->Defragment();
  }

  struct TmpBuffer {
    // Non-empty for buffers found in a PyArray_Storage. Multiple Arrays
    // can reference the same PjRtBuffer.
    std::vector<std::shared_ptr<PjRtBuffer>*> pjrt_buffer_ptrs;
    // TODO(skyewm): maybe use py_buffer's HostValue
    std::shared_ptr<Literal> host_copy;
  };

  // Synchronously copy all buffers to host
  absl::flat_hash_map<PjRtBuffer*, TmpBuffer> pjrt_buf_to_tmp_buffer;

  std::vector<PyArray> arrays = LiveArrays();
  for (const PyArray& array : arrays) {
    // TODO(hyeontaek): Support non-PjRt Arrays.
    // TODO(hyeontaek): Re-construct ifrt::Array with new PjRtBuffer so that
    // std::shared_ptr<PjRtBuffer> does not need to be updated in-place.
    if (array.ifrt_array() == nullptr) {
      continue;
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(array.ifrt_array());
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend "
          "only.");
    }
    TF_ASSIGN_OR_RETURN(absl::Span<std::shared_ptr<PjRtBuffer>> pjrt_buffers,
                        arr->mutable_pjrt_buffers());
    for (int i = 0; i < pjrt_buffers.size(); ++i) {
      std::shared_ptr<PjRtBuffer>& pjrt_buf_ptr = pjrt_buffers[i];
      if (pjrt_buf_ptr->IsDeleted()) {
        continue;
      }
      auto [iter, inserted] =
          pjrt_buf_to_tmp_buffer.insert({pjrt_buf_ptr.get(), TmpBuffer()});
      if (inserted) {
        TF_ASSIGN_OR_RETURN(iter->second.host_copy,
                            pjrt_buf_ptr->ToLiteralSync());
      }
      iter->second.pjrt_buffer_ptrs.push_back(&pjrt_buf_ptr);
    }
  }

  // All buffers successfully copied to host, delete on-device copies.
  //
  // Use blocking delete operation to ensure all memory is actually cleared
  // before we start rewriting buffers.
  //
  // Die instead of returning a bad status because program presumably can't
  // continue if we fail to reconstitute device buffers.
  for (const auto& it : pjrt_buf_to_tmp_buffer) {
    PjRtBuffer* pjrt_buf = it.first;
    TF_CHECK_OK(pjrt_buf
                    ->ReleaseDeviceMemoryOwnership(
                        /*wait_for_operations_to_complete=*/true)
                    .status());
  }

  // Copy host copies back to device and update PyArrays in-place.
  for (auto& it : pjrt_buf_to_tmp_buffer) {
    PjRtBuffer* pjrt_buf = it.first;
    TmpBuffer& tmp_buffer = it.second;
    std::unique_ptr<PjRtBuffer> new_copy =
        pjrt_client()
            ->BufferFromHostLiteral(*tmp_buffer.host_copy,
                                    pjrt_buf->memory_space())
            .value();
    TF_CHECK_OK(new_copy->GetReadyFuture().Await());

    std::shared_ptr<PjRtBuffer> new_pjrt_buf_ptr(new_copy.release());
    for (std::shared_ptr<PjRtBuffer>* pjrt_buffer_ptr :
         tmp_buffer.pjrt_buffer_ptrs) {
      *pjrt_buffer_ptr = new_pjrt_buf_ptr;
    }
  }

  // TODO(skyewm): delete executables?
  return absl::OkStatus();
}

/* static */ absl::StatusOr<nb::object> PyClient::BufferFromPyval(
    nb_class_ptr<PyClient> client, nb::handle argument, ifrt::Device* device,
    bool force_copy, ifrt::Client::HostBufferSemantics host_buffer_semantics) {
  if (device == nullptr) {
    TF_RET_CHECK(!client->ifrt_client_->addressable_devices().empty());
    device = client->ifrt_client_->addressable_devices().front();
  }
  CHECK(device != nullptr);

  auto transfer_guard_formatter = [&argument, dst_device = device] {
    auto type = nb::cast<std::string>(nb::str(argument.type()));
    // Catch exceptions because shape and dtype properties convertible to str
    // are not guaranteed to present in an arbitrary argument.
    std::string shape;
    std::string dtype;
    try {
      shape =
          nb::cast<std::string>(nb::str(nb::object(argument.attr("shape"))));
    } catch (const std::exception& e) {
      shape = "<unknown>";
    }
    try {
      dtype =
          nb::cast<std::string>(nb::str(nb::object(argument.attr("dtype"))));
    } catch (const std::exception& e) {
      dtype = "<unknown>";
    }
    return absl::StrCat("type=", type, ", shape=", shape, ", dtype=", dtype,
                        ", dst_device=", dst_device->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToHostToDevice(transfer_guard_formatter));

  TF_ASSIGN_OR_RETURN(ifrt::Device * found_device,
                      client->ifrt_client_->LookupDevice(device->Id()));
  if (found_device != device) {
    return InvalidArgument("Cannot copy value to device '%s' with '%s' backend",
                           device->DebugString(),
                           client->ifrt_client_->platform_name());
  }
  GlobalPyRefManager()->CollectGarbage();

  DevicePutOptions options;
  options.squash_64bit_types = false;
  options.allow_zero_copy =
      (!force_copy && (host_buffer_semantics ==
                       ifrt::Client::HostBufferSemantics::kImmutableZeroCopy));
  TF_ASSIGN_OR_RETURN(auto put_fn,
                      DevicePut(argument, client->ifrt_client_.get(), device,
                                options, ifrt::MemoryKind()));
  TF_ASSIGN_OR_RETURN(auto put, [&]() {
    // Must release the GIL before calling IFRT because backends may
    // decide to block/sleep for device buffer allocation.
    nb::gil_scoped_release gil_release;
    return std::move(put_fn)();
  }());

  if (put.ifrt_array) {
    auto traceback = Traceback::Get();
    return PyArray::MakeFromSingleDeviceArray(
        std::move(client), std::move(traceback), std::move(put.ifrt_array),
        /*weak_type=*/false,
        /*committed=*/false);
  } else {
    return put.owning_pybuffer;
  }
}

namespace {

// Makes IFRT `CompileOptions` from XLA `CompileOptions` and optional host
// callbacks.
std::unique_ptr<ifrt::CompileOptions> MakeIfrtCompileOptions(
    CompileOptions options, std::vector<nb::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that were
  // created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()` or
  // `PyClient::GetEmitPythonCallbackDescriptor()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(tsl::FormRef(
        static_cast<ifrt::LoadedHostCallback*>(host_callback.data())));
  }
  return std::make_unique<ifrt::XlaCompileOptions>(
      std::move(options), std::move(ifrt_loaded_host_callbacks));
}

// Makes IFRT `DeserializeExecutableOptions` from XLA `CompileOptions` and
// optional host callbacks.
std::unique_ptr<ifrt::DeserializeExecutableOptions>
MakeIfrtDeserializeExecutableOptions(std::optional<CompileOptions> options,
                                     std::vector<nb::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that were
  // created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()` or
  // `PyClient::GetEmitPythonCallbackDescriptor()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(tsl::FormRef(
        static_cast<ifrt::LoadedHostCallback*>(host_callback.data())));
  }
  return std::make_unique<ifrt::XlaDeserializeExecutableOptions>(
      std::move(options), std::move(ifrt_loaded_host_callbacks));
}

}  // namespace

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>>
PyClient::CompileIfrtProgram(
    nb_class_ptr<PyClient> client, std::unique_ptr<ifrt::Program> ifrt_program,
    std::unique_ptr<ifrt::CompileOptions> ifrt_options) {
  auto* pjrt_compatible_client =
      llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(
          client->ifrt_client_.get());
  auto* ifrt_xla_options =
      llvm::dyn_cast_or_null<ifrt::XlaCompileOptions>(ifrt_options.get());
  // For XLA programs, pass allocated device memory size to compile options for
  // pjrt compatible backends.
  if (pjrt_compatible_client != nullptr && ifrt_xla_options != nullptr) {
    xla::CompileOptions& options = ifrt_xla_options->compile_options;
    auto addressable_devices =
        pjrt_compatible_client->pjrt_client()->addressable_devices();
    if (!addressable_devices.empty()) {
      int device_ordinal = options.executable_build_options.device_ordinal();
      if (device_ordinal < 0) {
        device_ordinal = 0;
      }
      CHECK_LT(device_ordinal, addressable_devices.size());
      auto stats = addressable_devices[device_ordinal]->GetAllocatorStats();
      if (stats.ok() && stats->bytes_limit) {
        options.executable_build_options.set_device_memory_size(
            *stats->bytes_limit);
      }
    }

    if (pjrt_compatible_client->pjrt_client()->key_value_store().has_value()) {
      options.executable_build_options.set_key_value_store(
          *pjrt_compatible_client->pjrt_client()->key_value_store());
    }
  }

  std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable;
  std::optional<std::string> fingerprint;
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(ifrt_loaded_executable,
                        client->ifrt_client_->GetDefaultCompiler()->Compile(
                            std::move(ifrt_program), std::move(ifrt_options)));
    TF_RETURN_IF_ERROR(ifrt_loaded_executable->GetReadyFuture().Await());
    TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
  }
  auto traceback = Traceback::Get();
  return make_nb_class<PyLoadedExecutable>(
      std::move(client), std::move(ifrt_loaded_executable),
      std::move(traceback), std::move(fingerprint));
}

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>> PyClient::Compile(
    nb_class_ptr<PyClient> client, std::string mlir_module,
    CompileOptions options, std::vector<nb::capsule> host_callbacks) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(mlir_module, context));
  if (options.executable_build_options.use_shardy_partitioner()) {
    // Since Shardy is located in the middle of the XLA pipeline, we need to
    // export it before going to HLO while preserving Shardy ops and attrs.
    TF_RETURN_IF_ERROR(ExportShardyForHloRoundTrip(*module));
  }
  return CompileIfrtProgram(
      client, std::make_unique<xla::ifrt::HloProgram>(module.get()),
      MakeIfrtCompileOptions(std::move(options), std::move(host_callbacks)));
}

absl::StatusOr<nb::bytes> PyClient::SerializeExecutable(
    const PyLoadedExecutable& executable) const {
  TF_ASSIGN_OR_RETURN(auto serialized,
                      executable.ifrt_loaded_executable()->Serialize());
  return nb::bytes(serialized.data(), serialized.size());
}

/* static */ absl::StatusOr<nb_class_ptr<PyLoadedExecutable>>
PyClient::DeserializeExecutable(nb_class_ptr<PyClient> client,
                                nb::bytes serialized,
                                std::optional<CompileOptions> options,
                                std::vector<nb::capsule> host_callbacks) {
  std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable;
  std::optional<std::string> fingerprint;
  auto ifrt_deserialize_options = MakeIfrtDeserializeExecutableOptions(
      std::move(options), std::move(host_callbacks));
  {
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        ifrt_loaded_executable,
        client->ifrt_client_->GetDefaultCompiler()->DeserializeLoadedExecutable(
            absl::string_view(serialized.c_str(), serialized.size()),
            std::move(ifrt_deserialize_options)));
  }
  TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
  auto traceback = Traceback::Get();
  return make_nb_class<PyLoadedExecutable>(
      std::move(client), std::move(ifrt_loaded_executable),
      std::move(traceback), std::move(fingerprint));
}

namespace {

struct HeapProfileKey {
  Traceback* traceback;
  int64_t size;
  xla::PjRtDevice* device;
  bool operator==(const HeapProfileKey& other) const;
};

bool HeapProfileKey::operator==(const HeapProfileKey& other) const {
  if (size != other.size || device != other.device) {
    return false;
  }
  if ((traceback == nullptr) != (other.traceback == nullptr)) {
    return false;
  }
  if (traceback && traceback->raw_frames() != other.traceback->raw_frames()) {
    return false;
  }
  return true;
}

template <typename H>
H AbslHashValue(H h, const HeapProfileKey& key) {
  if (key.traceback) {
    h = H::combine(std::move(h), key.traceback->raw_frames());
  }
  h = H::combine(std::move(h), key.size, key.device);
  return h;
}

}  // namespace

absl::StatusOr<nb::bytes> PyClient::HeapProfile() {
  CHECK(PyGILState_Check());
  absl::flat_hash_set<PjRtBuffer*> buffer_set;
  absl::flat_hash_map<HeapProfileKey, int64_t> entries;

  auto add_buffer_to_profile = [&](PjRtBuffer* buffer, Traceback* traceback) {
    // We only wish to count each PjRtBuffer once, even though they may be
    // shared by multiple PyArrays.
    if (!buffer->IsDeleted() && buffer_set.insert(buffer).second) {
      TF_ASSIGN_OR_RETURN(size_t size, buffer->GetOnDeviceSizeInBytes());
      HeapProfileKey key{traceback, static_cast<int64_t>(size),
                         buffer->device()};
      ++entries[key];
    }
    return absl::OkStatus();
  };

  std::vector<PyArray> arrays = LiveArrays();
  for (const PyArray& array : arrays) {
    if (array.ifrt_array() == nullptr) {
      continue;
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(array.ifrt_array());
    // TODO(hyeontaek): Support non-PjRt Arrays.
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend "
          "only.");
    }
    for (const auto& buffer : arr->pjrt_buffers()) {
      TF_RETURN_IF_ERROR(add_buffer_to_profile(
          buffer.get(),
          array.traceback() ? array.traceback()->get() : nullptr));
    }
  }

  for (PyLoadedExecutable* executable = executables_; executable;
       executable = executable->next_) {
    if (!executable->is_deleted()) {
      HeapProfileKey key{
          executable->traceback() ? executable->traceback()->get() : nullptr,
          executable->SizeOfGeneratedCodeInBytes(), nullptr};
      ++entries[key];
    }
  }

  PprofProfileBuilder builder;
  auto* allocations = builder.profile().add_sample_type();
  allocations->set_type(builder.StringId("allocations"));
  allocations->set_unit(builder.StringId("count"));
  auto* space = builder.profile().add_sample_type();
  space->set_type(builder.StringId("space"));
  space->set_unit(builder.StringId("bytes"));

  const int kind_string_id = builder.StringId("kind");
  const int buffer_string_id = builder.StringId("buffer");
  const int executable_string_id = builder.StringId("executable");
  const int device_string_id = builder.StringId("device");
  for (const auto& entry : entries) {
    auto* sample = builder.profile().add_sample();
    if (entry.first.traceback) {
      for (const auto& frame : entry.first.traceback->raw_frames()) {
        sample->add_location_id(builder.LocationId(frame.first, frame.second));
      }
    }
    sample->add_value(entry.second);
    sample->add_value(entry.first.size * entry.second);

    auto* kind_label = sample->add_label();
    kind_label->set_key(kind_string_id);
    if (entry.first.device) {
      kind_label->set_str(buffer_string_id);
      auto* device_label = sample->add_label();
      device_label->set_key(device_string_id);
      std::string device_label_str(entry.first.device->DebugString());
      device_label->set_str(builder.StringId(device_label_str));
    } else {
      kind_label->set_str(executable_string_id);
    }
  }
  std::string serialized = builder.profile().SerializeAsString();
  return nb::bytes(serialized.data(), serialized.size());
}

absl::StatusOr<nb::object> PyClient::MakePythonCallbackUsingHostSendAndRecv(
    nb::callable callable, absl::Span<Shape const> operand_shapes,
    absl::Span<Shape const> result_shapes,
    absl::Span<uint16_t const> send_channel_ids,
    absl::Span<uint16_t const> recv_channel_ids, nb::callable serializer) {
  TF_ASSIGN_OR_RETURN(
      auto loaded_host_callback,
      PyHostSendAndRecvLoadedHostCallback::Create(
          ifrt_client(), std::move(callable), operand_shapes, result_shapes,
          send_channel_ids, recv_channel_ids, std::move(serializer)));
  nb::capsule callback_capsule(
      loaded_host_callback.release(), [](void* ptr) noexcept {
        static_cast<ifrt::LoadedHostCallback*>(ptr)->DropRef();
      });
  return callback_capsule;
}

// TODO(b/394595987): Remove this API method once we remove the call from
// mlir.py's get_emit_python_callback.
absl::StatusOr<std::pair<uint64_t, nb::object>>
PyClient::GetEmitPythonCallbackDescriptor(
    nb::callable callable, absl::Span<Shape const> operand_shapes,
    absl::Span<Shape const> result_shapes) {
  TF_ASSIGN_OR_RETURN(
      auto loaded_host_callback,
      PyCpuLoadedHostCallback::Create(ifrt_client(), std::move(callable),
                                      operand_shapes, result_shapes));
  const uint64_t descriptor = loaded_host_callback->descriptor();

  nb::capsule callback_capsule(
      loaded_host_callback.release(), [](void* ptr) noexcept {
        static_cast<ifrt::LoadedHostCallback*>(ptr)->DropRef();
      });
  return std::make_pair(descriptor, nb::object(std::move(callback_capsule)));
}

// TODO(b/394595987): Deprecate / clean up this API method to remove the need
// for `operand_shapes` and `result_shapes` once we can remove
// xla::PyClient::GetEmitPythonCallbackDescriptor (called by mlir.py's
// get_emit_python_callback for CPU/GPU devices).
absl::StatusOr<nb::object> PyClient::GetEmitPythonCallback(
    nb::callable callable) {
  absl::Span<const Shape> operand_shapes;
  absl::Span<const Shape> result_shapes;
  TF_ASSIGN_OR_RETURN(auto descriptor_and_callback,
                      GetEmitPythonCallbackDescriptor(
                          std::move(callable), operand_shapes, result_shapes));
  return nb::object(std::move(descriptor_and_callback.second));
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("xla_python_cpu_callback",
                                             &XlaPythonCpuCallback);

/* static */ int PyClient::tp_traverse(PyObject* self, visitproc visit,
                                       void* arg) {
  PyClient* c = nb::inst_ptr<PyClient>(self);
  for (const auto& [ifrt_device, py_device] : c->devices_) {
    Py_VISIT(py_device.ptr());
  }
  for (const auto& [ifrt_memory, py_memory] : c->memory_spaces_) {
    Py_VISIT(py_memory.ptr());
  }
  return 0;
}

/* static */ int PyClient::tp_clear(PyObject* self) {
  PyClient* c = nb::inst_ptr<PyClient>(self);
  absl::flat_hash_map<ifrt::Device*, nb_class_ptr<PyDevice>> devices;
  std::swap(devices, c->devices_);
  absl::flat_hash_map<ifrt::Memory*, nb_class_ptr<PyMemorySpace>> memory_spaces;
  std::swap(memory_spaces, c->memory_spaces_);
  return 0;
}

PyType_Slot PyClient::slots_[] = {
    {Py_tp_traverse, (void*)PyClient::tp_traverse},
    {Py_tp_clear, (void*)PyClient::tp_clear},
    {0, nullptr},
};

/* static */ void PyClient::RegisterPythonTypes(nb::module_& m) {
  nb::enum_<PjRtClient::HostBufferSemantics>(m, "HostBufferSemantics")
      .value("IMMUTABLE_ONLY_DURING_CALL",
             PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall)
      .value("IMMUTABLE_UNTIL_TRANSFER_COMPLETES",
             PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes)
      .value("ZERO_COPY", PjRtClient::HostBufferSemantics::kImmutableZeroCopy);

  nb::class_<PyClient> py_local_client(m, "Client", nb::is_weak_referenceable(),
                                       nb::type_slots(PyClient::slots_));
  py_local_client.def_prop_ro("platform", &PyClient::platform_name)
      .def_prop_ro("_raw_platform", &PyClient::raw_platform_name)
      .def_prop_ro("platform_version", &PyClient::platform_version)
      .def_prop_ro("runtime_type", &PyClient::runtime_type)
      .def("device_count", &PyClient::device_count)
      .def("local_device_count", &PyClient::addressable_device_count)
      .def("devices", &PyClient::Devices)
      .def("local_devices", &PyClient::LocalDevices)
      // TODO(hyeontaek): Remove this method once we have a unified API for
      // enumerating devices with different criteria.
      .def("_get_all_devices", &PyClient::GetAllDevices)
      .def("device_from_local_hardware_id",
           xla::ValueOrThrowWrapper(&PyClient::DeviceFromLocalHardwareId))
      .def("live_executables", &PyClient::LiveExecutables)
      .def("live_arrays", &PyClient::LiveArrays)
      .def("live_buffers", &PyClient::LiveArrays)
      .def("process_index", &PyClient::process_index)
      .def("host_id", &PyClient::process_index)
      .def("task_id", &PyClient::process_index)
      .def(
          "buffer_from_pyval",
          [](nb_class_ptr<PyClient> client, nb::handle argument,
             PyDevice* device, bool force_copy,
             PjRtClient::HostBufferSemantics host_buffer_semantics) {
            return ValueOrThrow(
                PyClient::BufferFromPyval(std::move(client), argument,
                                          device ? device->device() : nullptr,
                                          force_copy, host_buffer_semantics));
          },
          nb::arg("argument"), nb::arg("device").none() = nullptr,
          nb::arg("force_copy") = false,
          nb::arg("host_buffer_semantics") =
              PjRtClient::HostBufferSemantics::kImmutableZeroCopy)
      .def(
          "compile",
          [](nb_class_ptr<PyClient> client, nb::bytes mlir_module,
             CompileOptions options, std::vector<nb::capsule> host_callbacks) {
            return ValueOrThrow(PyClient::Compile(
                std::move(client),
                std::string(mlir_module.c_str(), mlir_module.size()),
                std::move(options), std::move(host_callbacks)));
          },
          nb::arg("computation"), nb::arg("compile_options") = CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>())
      .def(
          "compile",
          [](nb_class_ptr<PyClient> client, std::string mlir_module,
             CompileOptions options, std::vector<nb::capsule> host_callbacks) {
            return ValueOrThrow(PyClient::Compile(
                std::move(client), std::move(mlir_module), std::move(options),
                std::move(host_callbacks)));
          },
          nb::arg("computation"), nb::arg("compile_options") = CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>())
      .def("compile_ifrt_program",
           xla::ValueOrThrowWrapper(PyClient::CompileIfrtProgram))
      .def("serialize_executable",
           xla::ValueOrThrowWrapper(&PyClient::SerializeExecutable))
      .def(
          "deserialize_executable",
          [](nb_class_ptr<PyClient> client, nb::bytes serialized,
             std::optional<CompileOptions> options,
             std::vector<nb::capsule> host_callbacks) {
            return ValueOrThrow(PyClient::DeserializeExecutable(
                std::move(client), std::move(serialized), std::move(options),
                std::move(host_callbacks)));
          },
          nb::arg("serialized"), nb::arg("compile_options").none() = nb::none(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>())
      .def("heap_profile", xla::ValueOrThrowWrapper(&PyClient::HeapProfile))
      // TODO(zhangqiaorjc): Experimental.
      .def("defragment",
           [](PyClient& self) { xla::ThrowIfError(self.Defragment()); })
      .def("get_emit_python_callback_descriptor",
           xla::ValueOrThrowWrapper(&PyClient::GetEmitPythonCallbackDescriptor),
           nb::arg("callable"), nb::arg("operand_shapes"),
           nb::arg("result_shapes").none() = nb::none())
      .def("get_emit_python_callback",
           xla::ValueOrThrowWrapper(&PyClient::GetEmitPythonCallback),
           nb::arg("callable"))
      .def("make_python_callback_from_host_send_and_recv",
           xla::ValueOrThrowWrapper(
               &PyClient::MakePythonCallbackUsingHostSendAndRecv),
           nb::arg("callable"), nb::arg("operand_shapes"),
           nb::arg("result_shapes"), nb::arg("send_channel_ids"),
           nb::arg("recv_channel_ids"),
           nb::arg("serializer").none() = nb::none())
      .def(
          "get_default_layout",
          [](PyClient& self, nb_dtype dtype, nb::sequence shard_shape,
             nb_class_ptr<PyDevice> device)
              -> std::shared_ptr<const PjRtLayout> {
            ifrt::DType ifrt_type = xla::ValueOrThrow(DtypeToIfRtDType(dtype));
            std::vector<int64_t> dims = SequenceToVector<int64_t>(shard_shape);
            return xla::ValueOrThrow(self.ifrt_client()->GetDefaultLayout(
                ifrt_type, dims, device->device(), xla::ifrt::MemoryKind()));
          },
          nb::arg("dtype"), nb::arg("shard_shape"), nb::arg("device"))
      .def("__getattr__",
           [](PyClient& client, absl::string_view name) -> nb::object {
             const auto& attrs = client.Attributes().map();
             auto it = attrs.find(name);
             if (it != attrs.end()) {
               return std::visit([](auto&& v) { return nb::cast(v.value); },
                                 it->second);
             }
             throw nb::attribute_error(
                 absl::StrCat("Unknown attribute ", name).c_str());
           });
}

}  // namespace xla
