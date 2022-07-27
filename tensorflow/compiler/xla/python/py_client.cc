/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/py_client.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "tensorflow/compiler/xla/pjrt/host_callback.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/python/callback.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/pprof_profile_builder.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/core/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/python/py_client_gpu.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {

namespace py = pybind11;

PyClient::PyClient(std::unique_ptr<PjRtClient> pjrt_client)
    : PyClient(std::shared_ptr<PjRtClient>(std::move(pjrt_client))) {}

PyClient::PyClient(std::shared_ptr<PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)) {
  CHECK(pjrt_client_ != nullptr);
  buffers_.resize(pjrt_client_->device_count());
  for (PjRtDevice* device : pjrt_client_->addressable_devices()) {
    if (device->id() >= buffers_.size()) {
      buffers_.resize(device->id() + 1);
    }
  }
}

PyClient::~PyClient() {
  py::gil_scoped_release gil;
  pjrt_client_ = nullptr;
}

std::vector<ClientAndPtr<PjRtDevice>> PyClient::Devices() {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
  auto span = pjrt_client_->devices();
  devices.reserve(span.size());
  for (PjRtDevice* device : span) {
    devices.push_back(WrapWithClient(shared_from_this(), device));
  }
  return devices;
}

std::vector<ClientAndPtr<PjRtDevice>> PyClient::LocalDevices() {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
  devices.reserve(pjrt_client_->addressable_devices().size());
  for (PjRtDevice* device : pjrt_client_->addressable_devices()) {
    devices.push_back(WrapWithClient(shared_from_this(), device));
  }
  return devices;
}

std::vector<py::object> PyClient::LiveBuffers() {
  CHECK(PyGILState_Check());
  std::vector<py::object> buffers;
  for (PyBuffer* device_buffers : buffers_) {
    for (PyBuffer* buffer = device_buffers; buffer; buffer = buffer->next_) {
      if (!buffer->is_deleted()) {
        buffers.push_back(
            py::reinterpret_borrow<py::object>(buffer->AsHandle()));
      }
    }
  }
  return buffers;
}

std::vector<py::object> PyClient::LiveBuffersOnDevice(PjRtDevice* device) {
  CHECK_EQ(device->client(), pjrt_client());
  CHECK(PyGILState_Check());
  std::vector<py::object> buffers;
  for (PyBuffer* buffer = buffers_[device->id()]; buffer;
       buffer = buffer->next_) {
    if (!buffer->is_deleted()) {
      buffers.push_back(py::reinterpret_borrow<py::object>(buffer->AsHandle()));
    }
  }
  return buffers;
}

std::vector<std::shared_ptr<PyExecutable>> PyClient::LiveExecutables() {
  CHECK(PyGILState_Check());
  std::vector<std::shared_ptr<PyExecutable>> executables;
  for (PyExecutable* exec = executables_; exec; exec = exec->next_) {
    if (!exec->is_deleted()) {
      executables.push_back(exec->shared_from_this());
    }
  }
  return executables;
}

Status PyClient::Defragment() {
  CHECK(PyGILState_Check());
  switch (pjrt_client_->runtime_type()) {
    case PjRtRuntimeType::kTfrt:
      return pjrt_client_->Defragment();
    case PjRtRuntimeType::kStreamExecutor:
      struct TmpBuffer {
        PyBuffer* py_buffer;
        // TODO(skyewm): maybe use py_buffer's HostValue
        std::shared_ptr<Literal> host_copy;
      };

      // Synchronously copy all buffers to host
      std::vector<TmpBuffer> tmp_buffers;
      for (PyBuffer* device_buffers : buffers_) {
        for (PyBuffer* buffer = device_buffers; buffer;
             buffer = buffer->next_) {
          if (!buffer->is_deleted()) {
            TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                                buffer->buffer_->ToLiteralSync());
            tmp_buffers.push_back({buffer, literal});
          }
        }
      }

      // All buffers successfully copied to host, delete on-device copies.
      //
      // Use blocking delete operation to ensure all memory is actually cleared
      // before we start rewriting buffers.
      //
      // Die instead of returning a bad status because program presumably can't
      // continue if we fail to reconstitute device buffers.
      for (TmpBuffer& tmp_buffer : tmp_buffers) {
        TF_CHECK_OK(tensorflow::down_cast<PjRtStreamExecutorBuffer*>(
                        tmp_buffer.py_buffer->buffer_.get())
                        ->Release(/*wait_for_operations_to_complete=*/true)
                        .status());
      }

      // Copy host copies back to device and update PyBuffers in-place.
      for (TmpBuffer& tmp_buffer : tmp_buffers) {
        std::unique_ptr<PjRtBuffer> new_copy =
            pjrt_client_
                ->BufferFromHostLiteral(*tmp_buffer.host_copy,
                                        tmp_buffer.py_buffer->buffer_->device())
                .ValueOrDie();
        TF_CHECK_OK(new_copy->BlockHostUntilReady());
        tmp_buffer.py_buffer->buffer_.reset(new_copy.release());
      }

      // TODO(skyewm): delete executables?
  }
  return OkStatus();
}

StatusOr<std::vector<std::vector<ClientAndPtr<PjRtDevice>>>>
PyClient::GetDefaultDeviceAssignment(int num_replicas, int num_partitions) {
  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      pjrt_client_->GetDefaultDeviceAssignment(num_replicas, num_partitions));
  std::vector<std::vector<ClientAndPtr<PjRtDevice>>> result;
  result.resize(num_replicas);
  for (int r = 0; r < num_replicas; ++r) {
    result[r].resize(num_partitions);
    for (int p = 0; p < num_partitions; ++p) {
      int device_id = device_assignment(r, p);
      TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                          pjrt_client_->LookupDevice(device_id));
      result[r][p] = WrapWithClient(shared_from_this(), device);
    }
  }
  return result;
}

StatusOr<std::vector<ClientAndPtr<PjRtDevice>>>
PyClient::GetDefaultDeviceAssignment1D(int num_replicas) {
  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      pjrt_client_->GetDefaultDeviceAssignment(
                          num_replicas, /*num_partitions=*/1));
  std::vector<ClientAndPtr<PjRtDevice>> result;
  for (int i = 0; i < num_replicas; ++i) {
    int device_id = device_assignment(i, 0);
    TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                        pjrt_client_->LookupDevice(device_id));
    result.push_back(WrapWithClient(shared_from_this(), device));
  }
  return result;
}

StatusOr<py::object> PyClient::BufferFromPyval(
    pybind11::handle argument, PjRtDevice* device, bool force_copy,
    PjRtClient::HostBufferSemantics host_buffer_semantics) {
  if (device == nullptr) {
    TF_RET_CHECK(!pjrt_client_->addressable_devices().empty());
    device = pjrt_client_->addressable_devices().front();
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
                      pjrt_client_->LookupDevice(device->id()));
  if (found_device != device) {
    return InvalidArgument("Cannot copy value to device '%s' with '%s' backend",
                           device->DebugString(),
                           pjrt_client_->platform_name());
  }
  GlobalPyRefManager()->CollectGarbage();

  DevicePutOptions options;
  options.squash_64bit_types = false;
  options.allow_zero_copy =
      (!force_copy &&
       (host_buffer_semantics == PjRtClient::HostBufferSemantics::kZeroCopy));
  TF_ASSIGN_OR_RETURN(DevicePutResult put,
                      DevicePut(argument, device, options));

  if (put.owned_buffer) {
    auto traceback = Traceback::Get();
    return PyBuffer::Make(shared_from_this(), std::move(put.owned_buffer),
                          std::move(traceback));
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
      auto buffers, pjrt_client_->MakeCrossHostReceiveBuffers(
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
    auto traceback = Traceback::Get();
    auto py_buf =
        PyBuffer::Make(shared_from_this(), std::move(buffers[i]), traceback);
    result.push_back(std::make_pair(std::move(py_desc), std::move(py_buf)));
  }
  return result;
}

StatusOr<std::shared_ptr<PyExecutable>> PyClient::Compile(
    const XlaComputation& computation, CompileOptions options,
    std::vector<pybind11::capsule> host_callbacks) {
  std::unique_ptr<PjRtLoadedExecutable> executable;
  std::optional<std::string> fingerprint;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(executable,
                        pjrt_client_->Compile(computation, std::move(options)));
    TF_ASSIGN_OR_RETURN(fingerprint,
                        pjrt_client_->ExecutableFingerprint(*executable));
  }
  auto traceback = Traceback::Get();
  return std::make_shared<PyExecutable>(
      shared_from_this(), std::move(executable), std::move(traceback),
      std::move(fingerprint), std::move(host_callbacks));
}

StatusOr<std::shared_ptr<PyExecutable>> PyClient::CompileMlir(
    std::string mlir_module, CompileOptions options,
    std::vector<pybind11::capsule> host_callbacks) {
  std::unique_ptr<PjRtLoadedExecutable> executable;
  std::optional<std::string> fingerprint;
  {
    py::gil_scoped_release gil_release;
    mlir::MLIRContext context;
    TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                        ParseMlirModuleString(mlir_module, context));
    TF_ASSIGN_OR_RETURN(
        executable, pjrt_client_->Compile(module.get(), std::move(options)));
    TF_ASSIGN_OR_RETURN(fingerprint,
                        pjrt_client_->ExecutableFingerprint(*executable));
  }
  auto traceback = Traceback::Get();
  return std::make_shared<PyExecutable>(
      shared_from_this(), std::move(executable), std::move(traceback),
      std::move(fingerprint), std::move(host_callbacks));
}

StatusOr<py::bytes> PyClient::SerializeExecutable(
    const PyExecutable& executable) const {
  return pjrt_client_->SerializeExecutable(executable.pjrt_executable());
}

StatusOr<std::shared_ptr<PyExecutable>> PyClient::DeserializeExecutable(
    const std::string& serialized, CompileOptions options,
    std::vector<pybind11::capsule> host_callbacks) {
  std::unique_ptr<PjRtLoadedExecutable> executable;
  std::optional<std::string> fingerprint;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(executable, pjrt_client_->DeserializeExecutable(
                                        serialized, std::move(options)));
    TF_ASSIGN_OR_RETURN(fingerprint,
                        pjrt_client_->ExecutableFingerprint(*executable));
  }
  auto traceback = Traceback::Get();
  return std::make_shared<PyExecutable>(
      shared_from_this(), std::move(executable), std::move(traceback),
      std::move(fingerprint), std::move(host_callbacks));
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
  for (PyBuffer* device_buffers : buffers_) {
    for (PyBuffer* buffer = device_buffers; buffer; buffer = buffer->next_) {
      // We only wish to count each PjRtBuffer once, even though they may be
      // shared by multiple PyBuffers.
      if (!buffer->is_deleted() && buffer_set.insert(buffer->buffer()).second) {
        TF_ASSIGN_OR_RETURN(size_t size,
                            buffer->buffer()->GetOnDeviceSizeInBytes());
        HeapProfileKey key{buffer->traceback().get(),
                           static_cast<int64_t>(size),
                           buffer->buffer()->device()};
        ++entries[key];
      }
    }
  }

  for (PyExecutable* executable = executables_; executable;
       executable = executable->next_) {
    if (!executable->is_deleted()) {
      HeapProfileKey key{executable->traceback(),
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
          builder.StringId(entry.first.device->DebugString()));
    } else {
      kind_label->set_str(executable_string_id);
    }
  }
  return py::bytes(builder.profile().SerializeAsString());
}

namespace {

StatusOr<std::vector<CpuCallback::Arg>> CreateCallbackArgs(
    absl::Span<Shape const> operand_shapes) {
  std::vector<CpuCallback::Arg> callback_args(operand_shapes.size());
  for (int i = 0; i < operand_shapes.size(); ++i) {
    Shape shape = operand_shapes[i];

    if (shape.IsArray()) {
      Shape layout =
          (shape.has_layout() ? shape
                              : LayoutUtil::GetWithDefaultLayout(shape));
      callback_args[i].dims.resize(shape.dimensions_size());
      absl::c_copy(shape.dimensions(), callback_args[i].dims.begin());
      callback_args[i].strides = ByteStridesForShape(layout);
      callback_args[i].type = shape.element_type();
      callback_args[i].size_in_bytes = ShapeUtil::ByteSizeOf(layout);
      TF_ASSIGN_OR_RETURN(callback_args[i].dtype,
                          PrimitiveTypeToDtype(shape.element_type()));
    } else if (shape.IsToken()) {
      callback_args[i].type = TOKEN;
    } else {
      return InvalidArgument(
          "Only array and token arguments to Python callbacks are supported, "
          "got %s",
          shape.ToString());
    }
  }
  return callback_args;
}

StatusOr<std::vector<CpuCallback::Result>> CreateCallbackResults(
    absl::Span<Shape const> result_shapes) {
  std::vector<CpuCallback::Result> callback_results(result_shapes.size());
  for (int i = 0; i < result_shapes.size(); ++i) {
    if (result_shapes[i].IsArray()) {
      const Shape& shape =
          result_shapes[i].has_layout()
              ? result_shapes[i]
              : LayoutUtil::GetWithDefaultLayout(result_shapes[i]);
      callback_results[i].expected_dims.resize(shape.dimensions_size());
      absl::c_copy(shape.dimensions(),
                   callback_results[i].expected_dims.begin());
      callback_results[i].expected_strides = ByteStridesForShapeInt64(shape);
      callback_results[i].type = shape.element_type();
      callback_results[i].size_in_bytes = ShapeUtil::ByteSizeOf(shape);
      callback_results[i].reversed_layout.resize(shape.dimensions_size());
      absl::c_reverse_copy(shape.layout().minor_to_major(),
                           callback_results[i].reversed_layout.begin());
    } else if (result_shapes[i].IsToken()) {
      callback_results[i].type = TOKEN;
    } else {
      return InvalidArgument(
          "Only array and token return values from Python callbacks are "
          "supported, got %s",
          result_shapes[i].ToString());
    }
  }
  return callback_results;
}

}  // namespace

StatusOr<pybind11::object> PyClient::MakePythonCallbackUsingHostSendAndRecv(
    pybind11::function callable, absl::Span<Shape const> operand_shapes,
    absl::Span<Shape const> result_shapes,
    absl::Span<uint16_t const> send_channel_ids,
    absl::Span<uint16_t const> recv_channel_ids) {
  static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
                "Expected 64-bit pointers");

  TF_ASSIGN_OR_RETURN(auto callback_args, CreateCallbackArgs(operand_shapes));
  TF_ASSIGN_OR_RETURN(auto callback_results,
                      CreateCallbackResults(result_shapes));

  auto callback = std::make_shared<CpuCallback>(
      std::move(callable), callback_args, callback_results);

  auto* host_callback = new HostCallback;

  auto assign_arg_info = [](absl::Span<Shape const> shapes,
                            absl::Span<uint16_t const> channel_ids,
                            std::vector<HostCallbackArgInfo>& arg_infos) {
    DCHECK_EQ(shapes.size(), channel_ids.size());
    arg_infos.reserve(shapes.size());
    for (int i = 0; i < shapes.size(); ++i) {
      HostCallbackArgInfo host_callback_arg_info;
      host_callback_arg_info.channel_id = channel_ids[i];
      const auto& shape = shapes[i];
      Shape layout =
          (shape.has_layout() ? shape
                              : LayoutUtil::GetWithDefaultLayout(shape));
      host_callback_arg_info.shape = layout;
      arg_infos.push_back(std::move(host_callback_arg_info));
    }
  };

  assign_arg_info(operand_shapes, send_channel_ids, host_callback->operands);
  assign_arg_info(result_shapes, recv_channel_ids, host_callback->results);

  host_callback->callback = [callback = std::move(callback)](void** outputs,
                                                             void** inputs) {
    return callback->PrepareAndCall(outputs, inputs);
  };

  py::capsule callback_capsule(
      host_callback, [](void* ptr) { delete static_cast<HostCallback*>(ptr); });

  return callback_capsule;
}

StatusOr<std::pair<uint64_t, pybind11::object>>
PyClient::GetEmitPythonCallbackDescriptor(
    pybind11::function callable, absl::Span<Shape const> operand_shapes,
    absl::Span<Shape const> result_shapes) {
  PjRtPlatformId platform_id = pjrt_client_->platform_id();
  if (platform_id != GpuId() && platform_id != CpuId()) {
    return Unimplemented(
        "EmitPythonCallback is only implemented on CPU and GPU");
  }

  static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
                "Expected 64-bit pointers");

  TF_ASSIGN_OR_RETURN(auto callback_args, CreateCallbackArgs(operand_shapes));
  TF_ASSIGN_OR_RETURN(auto callback_results,
                      CreateCallbackResults(result_shapes));

  auto callback = std::make_unique<CpuCallback>(
      std::move(callable), callback_args, callback_results);
  uint64_t descriptor = absl::bit_cast<std::uint64_t>(callback.get());

  py::capsule callback_capsule(callback.release(), [](void* ptr) {
    delete reinterpret_cast<CpuCallback*>(ptr);
  });
  return std::make_pair(descriptor, py::object(std::move(callback_capsule)));
}

StatusOr<XlaOp> PyClient::EmitPythonCallbackFromDescriptor(
    XlaBuilder& builder, uint64_t descriptor, absl::Span<XlaOp const> operands,
    absl::Span<Shape const> result_shapes,
    std::optional<std::vector<Shape>> operand_layouts, bool has_side_effect) {
  std::vector<Shape> custom_call_arg_layouts(operands.size() + 1);
  custom_call_arg_layouts[0] =
      ShapeUtil::MakeShapeWithDescendingLayout(U64, {});
  std::vector<XlaOp> custom_call_args(operands.size() + 1);
  custom_call_args[0] = ConstantR0<std::uint64_t>(&builder, descriptor);
  absl::c_copy(operands, custom_call_args.begin() + 1);

  if (operand_layouts && operand_layouts->size() != operands.size()) {
    return InvalidArgument(
        "Mismatched number of operands (%d) and operand_layouts (%d)",
        operands.size(), operand_layouts->size());
  }

  for (int i = 0; i < operands.size(); ++i) {
    TF_ASSIGN_OR_RETURN(Shape shape, builder.GetShape(operands[i]));
    Shape layout = LayoutUtil::GetWithDefaultLayout(shape);
    if (shape.IsArray() && operand_layouts) {
      if (!(*operand_layouts)[i].has_layout()) {
        return InvalidArgument(
            "operand_layout shapes for callback must have "
            "layouts, got %s",
            (*operand_layouts)[i].ToString(/*print_layout=*/true));
      }
      if (!ShapeUtil::Compatible(shape, (*operand_layouts)[i])) {
        return InvalidArgument(
            "Incompatible shapes for Python callback argument %d: %s vs %s", i,
            shape.ToString(),
            (*operand_layouts)[i].ToString(/*print_layout=*/true));
      }
      layout = (*operand_layouts)[i];
    }
    custom_call_arg_layouts[i + 1] = layout;
  }

  std::vector<Shape> result_shapes_with_layout(result_shapes.size());
  for (int i = 0; i < result_shapes.size(); ++i) {
    if (result_shapes[i].IsArray()) {
      result_shapes_with_layout[i] =
          result_shapes[i].has_layout()
              ? result_shapes[i]
              : LayoutUtil::GetWithDefaultLayout(result_shapes[i]);
    } else if (result_shapes[i].IsToken()) {
      result_shapes_with_layout[i] = result_shapes[i];
    } else {
      return InvalidArgument(
          "Only array and token return values from Python callbacks are "
          "supported, got %s",
          result_shapes[i].ToString());
    }
  }
  custom_call_args[0] = ConstantR0<std::uint64_t>(&builder, descriptor);
  Shape result_shape = ShapeUtil::MakeTupleShape(result_shapes_with_layout);
  std::string callback_str = std::to_string(descriptor);
  std::string callback_name = "xla_python_cpu_callback";
  if (pjrt_client_->platform_id() == GpuId()) {
    callback_name = "xla_python_gpu_callback";
  }
  XlaOp result =
      CustomCallWithLayout(&builder, callback_name, custom_call_args,
                           result_shape, custom_call_arg_layouts,
                           /*opaque=*/callback_str.data(), has_side_effect,
                           /*output_operand_aliasing=*/{},
                           /*literal=*/nullptr,
                           /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
                           /*api_version=*/API_VERSION_STATUS_RETURNING);
  return result;
}

StatusOr<std::pair<XlaOp, pybind11::object>> PyClient::EmitPythonCallback(
    pybind11::function callable, XlaBuilder& builder,
    absl::Span<XlaOp const> operands, absl::Span<Shape const> result_shapes,
    std::optional<std::vector<Shape>> operand_layouts, bool has_side_effect) {
  std::vector<Shape> operand_shapes(operands.size());
  for (int i = 0; i < operands.size(); ++i) {
    TF_ASSIGN_OR_RETURN(Shape shape, builder.GetShape(operands[i]));
    operand_shapes[i] =
        (operand_layouts ? (*operand_layouts)[i]
                         : LayoutUtil::GetWithDefaultLayout(shape));
  }
  StatusOr<std::pair<uint64_t, pybind11::object>> result_sor =
      GetEmitPythonCallbackDescriptor(callable, operand_shapes, result_shapes);
  TF_ASSIGN_OR_RETURN(auto result, result_sor);
  uint64_t descriptor = result.first;
  pybind11::object keepalive = result.second;
  TF_ASSIGN_OR_RETURN(XlaOp callback_op,
                      EmitPythonCallbackFromDescriptor(
                          builder, descriptor, operands, result_shapes,
                          operand_shapes, has_side_effect));
  return std::make_pair(callback_op, keepalive);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("xla_python_cpu_callback",
                                             &XlaPythonCpuCallback);

#if TENSORFLOW_USE_ROCM
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("xla_python_gpu_callback",
                                         &XlaPythonGpuCallback, "ROCM");
#elif defined(GOOGLE_CUDA)
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("xla_python_gpu_callback",
                                         &XlaPythonGpuCallback, "CUDA");
#endif  // TENSORFLOW_USE_ROCM

}  // namespace xla
