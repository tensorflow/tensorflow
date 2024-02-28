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

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pprof_profile_builder.h"
#include "xla/python/py_array.h"
#include "xla/python/py_executable.h"
#include "xla/python/py_host_callback.h"
#include "xla/python/py_values.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/traceback.h"
#include "xla/python/transfer_guard_lib.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/platform_util.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/python/py_client_gpu.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {

namespace py = pybind11;

PyClient::PyClient(std::shared_ptr<ifrt::Client> ifrt_client)
    : ifrt_client_(std::move(ifrt_client)),
      client_attributes_(ifrt_client_->attributes()) {
  CHECK(ifrt_client_);
}

PyClient::~PyClient() {
  py::gil_scoped_release gil;
  ifrt_client_ = nullptr;
}

std::vector<ClientAndPtr<PjRtDevice>> PyClient::Devices() {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
  auto span = ifrt_client_->devices();
  devices.reserve(span.size());
  for (PjRtDevice* device : span) {
    devices.push_back(WrapWithClient(shared_from_this(), device));
  }
  return devices;
}

std::vector<ClientAndPtr<PjRtDevice>> PyClient::LocalDevices() {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
  devices.reserve(ifrt_client_->addressable_devices().size());
  for (ifrt::Device* device : ifrt_client_->addressable_devices()) {
    devices.push_back(WrapWithClient(shared_from_this(), device));
  }
  return devices;
}

StatusOr<ClientAndPtr<PjRtDevice>> PyClient::DeviceFromLocalHardwareId(
    int local_hardware_id) {
  TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                      ifrt_client_->LookupAddressableDevice(local_hardware_id));
  return WrapWithClient(shared_from_this(), device);
}

std::vector<py::object> PyClient::LiveBuffers() {
  CHECK(PyGILState_Check());
  std::vector<py::object> buffers;
  for (py::object& array : LiveArrays()) {
    buffers.push_back(std::move(array));
  }
  return buffers;
}

std::vector<std::shared_ptr<PyLoadedExecutable>> PyClient::LiveExecutables() {
  CHECK(PyGILState_Check());
  std::vector<std::shared_ptr<PyLoadedExecutable>> executables;
  for (PyLoadedExecutable* exec = executables_; exec; exec = exec->next_) {
    if (!exec->is_deleted()) {
      executables.push_back(exec->shared_from_this());
    }
  }
  return executables;
}

Status PyClient::Defragment() {
  CHECK(PyGILState_Check());
  auto runtime_type = ifrt_client_->runtime_type();
  if (runtime_type == PjRtRuntimeTypeString(PjRtRuntimeType::kTfrt)) {
    return pjrt_client()->Defragment();
  } else if (runtime_type ==
             PjRtRuntimeTypeString(PjRtRuntimeType::kStreamExecutor)) {
    struct TmpBuffer {
      // Non-empty for buffers found in a PyArray_Storage. Multiple Arrays
      // can reference the same PjRtBuffer.
      std::vector<std::shared_ptr<PjRtBuffer>*> pjrt_buffer_ptrs;
      // TODO(skyewm): maybe use py_buffer's HostValue
      std::shared_ptr<Literal> host_copy;
    };

    // Synchronously copy all buffers to host
    absl::flat_hash_map<PjRtBuffer*, TmpBuffer> pjrt_buf_to_tmp_buffer;

    for (PyArray_Storage* array = arrays_; array; array = array->next) {
      // TODO(hyeontaek): Support non-PjRt Arrays.
      // TODO(hyeontaek): Re-construct ifrt::Array with new PjRtBuffer so that
      // std::shared_ptr<PjRtBuffer> does not need to be updated in-place.
      if (array->ifrt_array == nullptr) {
        continue;
      }
      auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(
          array->ifrt_array.get());
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
      TF_CHECK_OK(tensorflow::down_cast<PjRtStreamExecutorBuffer*>(pjrt_buf)
                      ->Release(/*wait_for_operations_to_complete=*/true)
                      .status());
    }

    // Copy host copies back to device and update PyArrays in-place.
    for (auto& it : pjrt_buf_to_tmp_buffer) {
      PjRtBuffer* pjrt_buf = it.first;
      TmpBuffer& tmp_buffer = it.second;
      std::unique_ptr<PjRtBuffer> new_copy =
          pjrt_client()
              ->BufferFromHostLiteral(*tmp_buffer.host_copy, pjrt_buf->device())
              .value();
      TF_CHECK_OK(new_copy->BlockHostUntilReady());

      std::shared_ptr<PjRtBuffer> new_pjrt_buf_ptr(new_copy.release());
      for (std::shared_ptr<PjRtBuffer>* pjrt_buffer_ptr :
           tmp_buffer.pjrt_buffer_ptrs) {
        *pjrt_buffer_ptr = new_pjrt_buf_ptr;
      }
    }

    // TODO(skyewm): delete executables?
  }
  return OkStatus();
}

StatusOr<py::object> PyClient::BufferFromPyval(
    pybind11::handle argument, PjRtDevice* device, bool force_copy,
    ifrt::Client::HostBufferSemantics host_buffer_semantics) {
  if (device == nullptr) {
    TF_RET_CHECK(!ifrt_client_->addressable_devices().empty());
    device = ifrt_client_->addressable_devices().front();
  }
  CHECK(device != nullptr);

  auto transfer_guard_formatter = [&argument, dst_device = device] {
    auto type = py::cast<std::string>(py::str(argument.get_type()));
    // Catch exceptions because shape and dtype properties convertible to str
    // are not guaranteed to present in an arbitrary argument.
    std::string shape;
    std::string dtype;
    try {
      shape = py::cast<std::string>(py::str(argument.attr("shape")));
    } catch (const std::exception& e) {
      shape = "<unknown>";
    }
    try {
      dtype = py::cast<std::string>(py::str(argument.attr("dtype")));
    } catch (const std::exception& e) {
      dtype = "<unknown>";
    }
    return absl::StrCat("type=", type, ", shape=", shape, ", dtype=", dtype,
                        ", dst_device=", dst_device->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToHostToDevice(transfer_guard_formatter));

  TF_ASSIGN_OR_RETURN(PjRtDevice * found_device,
                      ifrt_client_->LookupDevice(device->id()));
  if (found_device != device) {
    return InvalidArgument("Cannot copy value to device '%s' with '%s' backend",
                           device->DebugString(),
                           ifrt_client_->platform_name());
  }
  GlobalPyRefManager()->CollectGarbage();

  DevicePutOptions options;
  options.squash_64bit_types = false;
  options.allow_zero_copy =
      (!force_copy &&
       (host_buffer_semantics == ifrt::Client::HostBufferSemantics::kZeroCopy));
  TF_ASSIGN_OR_RETURN(DevicePutResult put,
                      DevicePut(argument, ifrt_client_.get(), device, options,
                                ifrt::MemoryKind()));

  if (put.ifrt_array) {
    auto traceback = Traceback::Get();
    return PyArray::MakeFromSingleDeviceArray(
        shared_from_this(), std::move(traceback), std::move(put.ifrt_array),
        /*weak_type=*/false,
        /*committed=*/false);
  } else {
    return py::reinterpret_borrow<py::object>(put.owning_pybuffer);
  }
}

StatusOr<std::vector<std::pair<pybind11::bytes, pybind11::object>>>
PyClient::MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                                      PjRtDevice* device) {
  CHECK(device != nullptr);
  absl::Mutex mu;
  StatusOr<std::vector<PjRtCrossHostRecvDescriptors>> recv_descriptors_or;
  bool done = false;

  TF_ASSIGN_OR_RETURN(
      auto buffers, pjrt_client()->MakeCrossHostReceiveBuffers(
                        shapes, device,
                        [&done, &recv_descriptors_or,
                         &mu](StatusOr<PjRtCrossHostRecvState> recv_state_or) {
                          absl::MutexLock l(&mu);
                          if (recv_state_or.ok()) {
                            py::gil_scoped_acquire gil;
                            recv_descriptors_or =
                                std::move(recv_state_or->descriptors);
                          } else {
                            recv_descriptors_or = recv_state_or.status();
                          }
                          done = true;
                        }));

  {
    py::gil_scoped_release gil_release;
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&done));
  }

  TF_RETURN_IF_ERROR(recv_descriptors_or.status());
  CHECK_EQ(buffers.size(), recv_descriptors_or->size());
  std::vector<std::pair<pybind11::bytes, pybind11::object>> result;
  result.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    auto& descriptors = recv_descriptors_or->at(i);
    CHECK_EQ(descriptors.serialized_descriptors.size(), 1);
    const std::string& desc = descriptors.serialized_descriptors[0];
    pybind11::bytes py_desc = pybind11::bytes(desc);
    auto* client =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(ifrt_client());
    if (client == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    TF_ASSIGN_OR_RETURN(auto ifrt_array,
                        client->CreatePjRtArray(std::move(buffers[i])));
    auto py_buf = PyArray::MakeFromSingleDeviceArray(
        shared_from_this(), Traceback::Get(), std::move(ifrt_array),
        /*weak_type=*/false,
        /*committed=*/false);
    result.push_back(std::make_pair(std::move(py_desc), std::move(py_buf)));
  }
  return result;
}

namespace {

// Makes IFRT `CompileOptions` from XLA `CompileOptions` and optional host
// callbacks.
std::unique_ptr<ifrt::CompileOptions> MakeIfrtCompileOptions(
    CompileOptions options, std::vector<pybind11::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that were
  // created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()` or
  // `PyClient::GetEmitPythonCallbackDescriptor()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(
        tsl::FormRef(host_callback.get_pointer<ifrt::LoadedHostCallback>()));
  }
  return std::make_unique<ifrt::XlaCompileOptions>(
      std::move(options), std::move(ifrt_loaded_host_callbacks));
}

// Makes IFRT `DeserializeExecutableOptions` from XLA `CompileOptions` and
// optional host callbacks.
std::unique_ptr<ifrt::DeserializeExecutableOptions>
MakeIfrtDeserializeExecutableOptions(
    std::optional<CompileOptions> options,
    std::vector<pybind11::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that were
  // created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()` or
  // `PyClient::GetEmitPythonCallbackDescriptor()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(
        tsl::FormRef(host_callback.get_pointer<ifrt::LoadedHostCallback>()));
  }
  return std::make_unique<ifrt::XlaDeserializeExecutableOptions>(
      std::move(options), std::move(ifrt_loaded_host_callbacks));
}

}  // namespace

StatusOr<std::shared_ptr<PyLoadedExecutable>> PyClient::Compile(
    std::string mlir_module, CompileOptions options,
    std::vector<pybind11::capsule> host_callbacks) {
  // Pass allocated device memory size to compile options for pjrt compatible
  // backends.
  auto* pjrt_compatible_client =
      llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(ifrt_client_.get());
  if (pjrt_compatible_client != nullptr) {
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
  }

  std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable;
  std::optional<std::string> fingerprint;
  auto ifrt_compile_options =
      MakeIfrtCompileOptions(std::move(options), std::move(host_callbacks));
  {
    py::gil_scoped_release gil_release;
    mlir::MLIRContext context;
    TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                        ParseMlirModuleString(mlir_module, context));
    TF_ASSIGN_OR_RETURN(
        ifrt_loaded_executable,
        ifrt_client_->GetDefaultCompiler()->Compile(
            std::make_unique<xla::ifrt::XlaProgram>(module.get()),
            std::move(ifrt_compile_options)));
    TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
  }
  auto traceback = Traceback::Get();
  return std::make_shared<PyLoadedExecutable>(
      shared_from_this(), std::move(ifrt_loaded_executable),
      std::move(traceback), std::move(fingerprint));
}

StatusOr<py::bytes> PyClient::SerializeExecutable(
    const PyLoadedExecutable& executable) const {
  return executable.ifrt_loaded_executable()->Serialize();
}

StatusOr<std::shared_ptr<PyLoadedExecutable>> PyClient::DeserializeExecutable(
    const std::string& serialized, std::optional<CompileOptions> options,
    std::vector<pybind11::capsule> host_callbacks) {
  std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable;
  std::optional<std::string> fingerprint;
  auto ifrt_deserialize_options = MakeIfrtDeserializeExecutableOptions(
      std::move(options), std::move(host_callbacks));
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        ifrt_loaded_executable,
        ifrt_client_->GetDefaultCompiler()->DeserializeLoadedExecutable(
            serialized, std::move(ifrt_deserialize_options)));
    TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
  }
  TF_ASSIGN_OR_RETURN(fingerprint, ifrt_loaded_executable->Fingerprint());
  auto traceback = Traceback::Get();
  return std::make_shared<PyLoadedExecutable>(
      shared_from_this(), std::move(ifrt_loaded_executable),
      std::move(traceback), std::move(fingerprint));
}

namespace {

struct HeapProfileKey {
  Traceback* traceback;
  int64_t size;
  PjRtDevice* device;
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

StatusOr<py::bytes> PyClient::HeapProfile() {
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
    return OkStatus();
  };

  for (PyArray_Storage* array = arrays_; array; array = array->next) {
    if (array->ifrt_array == nullptr) {
      continue;
    }
    auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(
        array->ifrt_array.get());
    // TODO(hyeontaek): Support non-PjRt Arrays.
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend "
          "only.");
    }
    for (const auto& buffer : arr->pjrt_buffers()) {
      TF_RETURN_IF_ERROR(add_buffer_to_profile(
          buffer.get(), array->traceback ? array->traceback->get() : nullptr));
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
      device_label->set_str(
          builder.StringId(std::string(entry.first.device->DebugString())));
    } else {
      kind_label->set_str(executable_string_id);
    }
  }
  return py::bytes(builder.profile().SerializeAsString());
}

StatusOr<pybind11::object> PyClient::MakePythonCallbackUsingHostSendAndRecv(
    pybind11::function callable, absl::Span<Shape const> operand_shapes,
    absl::Span<Shape const> result_shapes,
    absl::Span<uint16_t const> send_channel_ids,
    absl::Span<uint16_t const> recv_channel_ids,
    pybind11::function serializer) {
  TF_ASSIGN_OR_RETURN(
      auto loaded_host_callback,
      PyHostSendAndRecvLoadedHostCallback::Create(
          ifrt_client(), std::move(callable), operand_shapes, result_shapes,
          send_channel_ids, recv_channel_ids, std::move(serializer)));
  py::capsule callback_capsule(loaded_host_callback.release(), [](void* ptr) {
    static_cast<ifrt::LoadedHostCallback*>(ptr)->DropRef();
  });
  return callback_capsule;
}

StatusOr<std::pair<uint64_t, pybind11::object>>
PyClient::GetEmitPythonCallbackDescriptor(
    pybind11::function callable, absl::Span<Shape const> operand_shapes,
    absl::Span<Shape const> result_shapes) {
  TF_ASSIGN_OR_RETURN(
      auto loaded_host_callback,
      PyCpuLoadedHostCallback::Create(ifrt_client(), std::move(callable),
                                      operand_shapes, result_shapes));
  const uint64_t descriptor = loaded_host_callback->descriptor();

  py::capsule callback_capsule(loaded_host_callback.release(), [](void* ptr) {
    static_cast<ifrt::LoadedHostCallback*>(ptr)->DropRef();
  });
  return std::make_pair(descriptor, py::object(std::move(callback_capsule)));
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("xla_python_cpu_callback",
                                             &XlaPythonCpuCallback);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "xla_python_gpu_callback", &XlaPythonGpuCallback,
    absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value()));
#endif

}  // namespace xla
