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

#include "tensorflow/compiler/xla/python/py_executable.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#ifdef JAX_ENABLE_IFRT
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#endif
#include "tensorflow/compiler/xla/pjrt/host_callback.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/tsl/platform/fingerprint.h"

namespace xla {

namespace py = pybind11;

Status PyToken::Await() {
  CHECK(future_.IsValid());
  py::gil_scoped_release gil_release;
  return future_.Await();
}

Status PyShardedToken::Await() {
  py::gil_scoped_release gil_release;
  Status status = OkStatus();
  for (auto& future : futures_) {
    auto s = future.Await();
    if (!s.ok()) status = std::move(s);
  }
  return status;
}

PyLoadedExecutable::PyLoadedExecutable(
    std::shared_ptr<PyClient> client,
#ifdef JAX_ENABLE_IFRT
    std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable,
#else
    std::unique_ptr<PjRtLoadedExecutable> executable,
#endif
    std::shared_ptr<Traceback> traceback,
    std::optional<std::string> fingerprint,
    std::vector<pybind11::capsule> host_callbacks)
    : client_(std::move(client)),
#ifdef JAX_ENABLE_IFRT
      ifrt_loaded_executable_(std::move(ifrt_loaded_executable)),
#else
      executable_(std::move(executable)),
#endif
      traceback_(std::move(traceback)),
      fingerprint_(std::move(fingerprint)),
      host_callbacks_(std::move(host_callbacks)) {
  CHECK(PyGILState_Check());
  next_ = client_->executables_;
  client_->executables_ = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
  options_.untuple_result = true;
  if (fingerprint_) {
    options_.launch_id = tsl::Fingerprint32(*fingerprint_);
#ifdef JAX_ENABLE_IFRT
    VLOG(1) << "Fingerprint for executable " << ifrt_loaded_executable_->name()
            << ": " << *fingerprint_;
#else
    VLOG(1) << "Fingerprint for executable " << executable_->name() << ": "
            << *fingerprint_;
#endif
  }
}

PyLoadedExecutable::~PyLoadedExecutable() {
  CHECK(PyGILState_Check());
  if (client_->executables_ == this) {
    client_->executables_ = next_;
  }
  if (prev_) {
    prev_->next_ = next_;
  }
  if (next_) {
    next_->prev_ = prev_;
  }
}

std::vector<ClientAndPtr<PjRtDevice>> PyLoadedExecutable::AddressableDevices()
    const {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
#ifdef JAX_ENABLE_IFRT
  devices.reserve(ifrt_loaded_executable_->addressable_devices().size());
  for (ifrt::Device* device : ifrt_loaded_executable_->addressable_devices()) {
    devices.push_back(WrapWithClient(client_, device));
  }
#else
  devices.reserve(executable_->addressable_devices().size());
  for (PjRtDevice* device : executable_->addressable_devices()) {
    devices.push_back(WrapWithClient(client_, device));
  }
#endif
  return devices;
}

StatusOr<std::pair<std::vector<PyBuffer::object>, PyToken>>
PyLoadedExecutable::ExecuteInternal(
    absl::Span<PyBuffer::object const> args, PjRtDevice* device,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
#ifdef JAX_ENABLE_IFRT
  std::vector<std::unique_ptr<ifrt::Array>> output_arrays;
  std::unique_ptr<ifrt::Future<Status>> returned_future;
#else
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
#endif
  {
    auto options = options_;
    std::shared_ptr<HostCallbackStates> host_callback_states;

    if (!host_callbacks_.empty()) {
      auto* host_memory_for_device_manager =
          client()->pjrt_client()->GetPjRtHostMemoryForDeviceManager();
      if (host_memory_for_device_manager == nullptr) {
        return InternalError("Host callback not supported for runtime type: %s",
                             client()->runtime_type());
      }

      returned_futures.emplace();

      host_callback_states = std::make_shared<HostCallbackStates>();
      auto& contexts = host_callback_states->contexts.emplace_back();
      auto& send_callbacks =
          host_callback_states->send_callbacks.emplace_back();
      auto& recv_callbacks =
          host_callback_states->recv_callbacks.emplace_back();

      for (const py::capsule& host_callback : host_callbacks_) {
        contexts.push_back(CreateHostCallbackStateAndAppendSendRecvCallbacks(
            *host_callback.get_pointer<HostCallback>(),
            host_memory_for_device_manager, send_callbacks, recv_callbacks));
      }
      options.send_callbacks = host_callback_states->send_callbacks;
      options.recv_callbacks = host_callback_states->recv_callbacks;
    }

    py::gil_scoped_release gil_release;
#ifdef JAX_ENABLE_IFRT
    std::vector<ifrt::Array*> arg_arrays(args.size());
    absl::c_transform(
        args, arg_arrays.begin(),
        [](const PyBuffer::object& buf) { return buf.buf()->ifrt_array(); });
#else
    std::vector<PjRtBuffer*> arg_buffers(args.size());
    absl::c_transform(
        args, arg_buffers.begin(),
        [](const PyBuffer::object& buf) { return buf.buf()->pjrt_buffer(); });
#endif

#ifdef JAX_ENABLE_IFRT
    if (device) {
      TF_ASSIGN_OR_RETURN(
          auto result,
          ifrt_loaded_executable()->Execute(
              arg_arrays, options,
              /*devices=*/
              ifrt::DeviceList(ifrt::DeviceList::Devices({device}))));
      if (returned_futures.has_value()) {
        returned_futures->push_back(std::move(result.status));
      }
      output_arrays = std::move(result.outputs);
    } else {
      TF_ASSIGN_OR_RETURN(auto result, ifrt_loaded_executable()->Execute(
                                           arg_arrays, options,
                                           /*devices=*/std::nullopt));
      if (returned_futures.has_value()) {
        returned_futures->push_back(std::move(result.status));
      }
      output_arrays = std::move(result.outputs);
    }
#else
    if (device) {
      std::optional<PjRtFuture<Status>> future;
      output_buffers.resize(1);
      TF_ASSIGN_OR_RETURN(
          output_buffers[0],
          executable_->ExecutePortable(arg_buffers, device, options, future,
                                       returned_futures.has_value()));
      if (future) {
        returned_futures->emplace_back(std::move(*future));
      }
    } else {
      TF_ASSIGN_OR_RETURN(
          output_buffers,
          executable_->Execute({arg_buffers}, options, returned_futures));
    }
#endif

    if (!host_callbacks_.empty()) {
      // For host callbacks to work, `returned_futures` must not be nullopt.
      returned_futures->at(0).OnReady([host_callback_states](Status) mutable {
        host_callback_states.reset();
      });
    }
  }
  auto traceback = Traceback::Get();
  std::vector<PyBuffer::object> outputs;
#ifdef JAX_ENABLE_IFRT
  outputs.reserve(output_arrays.size());
  for (auto& array : output_arrays) {
    outputs.push_back(PyBuffer::Make(client_, std::move(array), traceback));
  }

  if (!returned_futures.has_value()) {
    return std::pair<std::vector<PyBuffer::object>, PyToken>(
        std::move(outputs), PyToken::ReadyPyToken());
  }
  return std::pair<std::vector<PyBuffer::object>, PyToken>(
      std::move(outputs), PyToken(std::move(returned_futures->at(0))));
#else
  outputs.reserve(output_buffers[0].size());
  for (auto& buffer : output_buffers[0]) {
    outputs.push_back(PyBuffer::Make(client_, std::move(buffer), traceback));
  }

  // TODO(b/240696624): Although the PjRt interface require `returned_futures`
  // to be resized correctly if it is not nullopt, some implementation does not
  // implement this. So we have to check whether returned_futures is empty.
  // Remove this check once the implementation is fixed.
  if (!returned_futures.has_value()) {
    return std::pair<std::vector<PyBuffer::object>, PyToken>(
        std::move(outputs), PyToken::ReadyPyToken());
  }
  return std::pair<std::vector<PyBuffer::object>, PyToken>(
      std::move(outputs), PyToken(std::move(returned_futures->at(0))));
#endif
}

StatusOr<std::pair<std::vector<PyBuffer::object>, PyToken>>
PyLoadedExecutable::ExecuteWithToken(absl::Span<PyBuffer::object const> args,
                                     PjRtDevice* device) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
#ifdef JAX_ENABLE_IFRT
  returned_futures.emplace();
#else
  if (executable_->IsReturnedFutureSupported()) returned_futures.emplace();
#endif
  return ExecuteInternal(args, device, returned_futures);
}

StatusOr<std::vector<PyBuffer::object>> PyLoadedExecutable::Execute(
    absl::Span<PyBuffer::object const> args, PjRtDevice* device) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  TF_ASSIGN_OR_RETURN(auto outputs_and_token,
                      ExecuteInternal(args, device, returned_futures));
  return std::move(outputs_and_token.first);
}

namespace {

// Traits classes of common methods for std::vector<PyBuffer::object> and
// PyShardedBuffer.
template <typename ShardedBufferT>
struct ShardedBufferAdapter;

template <>
struct ShardedBufferAdapter<PyShardedBuffer*> {
  using ResultT = PyShardedBuffer;
  static int num_devices(const PyShardedBuffer* arg) {
    DCHECK(arg);
    return arg->num_devices();
  }
#ifdef JAX_ENABLE_IFRT
  static ifrt::Array* GetIfRtArray(
      const PyShardedBuffer* arg,
      std::vector<std::unique_ptr<ifrt::Array>>& owned_ifrt_arrays) {
    DCHECK(arg);
    DCHECK(arg->ifrt_array());
    return arg->ifrt_array();
  }
#else
  static PjRtBuffer* GetPjRtBuffer(const PyShardedBuffer* arg, int device_id) {
    return arg->pjrt_buffer(device_id);
  }
#endif
};

template <>
struct ShardedBufferAdapter<std::vector<PyBuffer::object>> {
  using ResultT = std::vector<PyBuffer::object>;
  static int num_devices(const std::vector<PyBuffer::object>& arg) {
    return arg.size();
  }
#ifdef JAX_ENABLE_IFRT
  static ifrt::Array* GetIfRtArray(
      const std::vector<PyBuffer::object>& arg,
      std::vector<std::unique_ptr<ifrt::Array>>& owned_ifrt_arrays) {
    // TODO(hyeontaek): This on-demand Array creation is not efficient and has
    // insufficient information about the shape (a dummy shape is used). This
    // should be removed if possible and only be used in the context where the
    // shape information is unused.
    DCHECK(&arg);
    std::vector<ifrt::Array*> ifrt_arrays;
    ifrt_arrays.reserve(arg.size());
    ifrt::DeviceList::Devices devices;
    devices.reserve(arg.size());
    for (auto& buf : arg) {
      DCHECK(buf.buf());
      DCHECK(buf.buf()->ifrt_array());
      ifrt_arrays.push_back(buf.buf()->ifrt_array());
      devices.push_back(buf.buf()->ifrt_array()->sharding().devices().front());
      // Do not need to collect per-device shapes because the created array is
      // not supposed to explode.
    }
    CHECK(!ifrt_arrays.empty());
    // Use a dummy shape.
    // TODO(hyeontaek): Find a way to compute a correct shape.
    auto ifrt_array =
        ifrt_arrays.front()->client()->AssembleArrayFromSingleDeviceArrays(
            ifrt_arrays.front()->shape(),
            ifrt::OpaqueSharding::Create(ifrt::DeviceList(std::move(devices))),
            ifrt_arrays, ifrt::ArrayCopySemantics::kReuseInput);
    TF_CHECK_OK(ifrt_array.status());
    owned_ifrt_arrays.push_back(*std::move(ifrt_array));
    return owned_ifrt_arrays.back().get();
  }
#else
  static PjRtBuffer* GetPjRtBuffer(const std::vector<PyBuffer::object>& arg,
                                   int device_id) {
    return arg.at(device_id).buf()->pjrt_buffer();
  }
#endif
};

void PopulateExecuteShardedResults(
    const std::shared_ptr<PyClient>& client,
#ifdef JAX_ENABLE_IFRT
    std::vector<std::unique_ptr<ifrt::Array>> ifrt_arrays, int num_computations,
#else
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> pjrt_buffers,
#endif
    std::vector<PyShardedBuffer>& outputs) {
  auto traceback = Traceback::Get();
#ifdef JAX_ENABLE_IFRT
  int num_output_buffers = ifrt_arrays.size();
#else
  int num_computations = pjrt_buffers.size();
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = pjrt_buffers[0].size();
#endif
  outputs.reserve(num_output_buffers);
#ifdef JAX_ENABLE_IFRT
  for (auto& array : ifrt_arrays) {
    outputs.emplace_back(client, std::move(array), traceback);
  }
#else
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    std::vector<std::shared_ptr<PjRtBuffer>> buffers;
    buffers.reserve(num_computations);
    for (int computation = 0; computation < num_computations; ++computation) {
      buffers.push_back(std::move(pjrt_buffers[computation][buffer_id]));
    }
    outputs.emplace_back(client, std::move(buffers), traceback);
  }
#endif
}

void PopulateExecuteShardedResults(
    const std::shared_ptr<PyClient>& client,
#ifdef JAX_ENABLE_IFRT
    std::vector<std::unique_ptr<ifrt::Array>> ifrt_arrays, int num_computations,
#else
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> pjrt_buffers,
#endif
    std::vector<std::vector<PyBuffer::object>>& outputs) {
  auto traceback = Traceback::Get();
#ifdef JAX_ENABLE_IFRT
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = ifrt_arrays.size();
#else
  int num_computations = pjrt_buffers.size();
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = pjrt_buffers[0].size();
#endif
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    outputs[buffer_id].reserve(num_computations);
#ifdef JAX_ENABLE_IFRT
    auto exploded_arrays =
        ifrt_arrays[buffer_id]->DisassembleIntoSingleDeviceArrays(
            ifrt::ArrayCopySemantics::kReuseInput);
    TF_CHECK_OK(exploded_arrays.status());
    for (auto& exploded_array : *exploded_arrays) {
      outputs[buffer_id].push_back(
          PyBuffer::Make(client, std::move(exploded_array), traceback));
    }
#else
    for (int computation = 0; computation < num_computations; ++computation) {
      outputs[buffer_id].push_back(PyBuffer::Make(
          client, std::move(pjrt_buffers[computation][buffer_id]), traceback));
    }
#endif
  }
}

template <typename ArgT,
          typename ResultT = typename ShardedBufferAdapter<ArgT>::ResultT,
          typename ArgAdapter = ShardedBufferAdapter<ArgT>>
StatusOr<std::pair<std::vector<ResultT>, PyShardedToken>>
ExecuteShardedOnLocalDevicesInternal(
    const ExecuteOptions& options, const std::shared_ptr<PyClient>& client,
#ifdef JAX_ENABLE_IFRT
    ifrt::LoadedExecutable* ifrt_loaded_executable,
#else
    PjRtLoadedExecutable* executable,
#endif
    absl::Span<const py::capsule> host_callbacks, absl::Span<const ArgT> args,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
#ifdef JAX_ENABLE_IFRT
  std::vector<std::unique_ptr<ifrt::Array>> output_arrays;
  std::unique_ptr<ifrt::Future<Status>> returned_future;
  int num_computations = ifrt_loaded_executable->addressable_devices().size();
#else
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  int num_computations = executable->addressable_devices().size();
#endif
  {
    auto opts = options;
    std::shared_ptr<HostCallbackStates> host_callback_states;
    if (!host_callbacks.empty()) {
      auto* host_memory_for_device_manager =
          client->pjrt_client()->GetPjRtHostMemoryForDeviceManager();
      if (host_memory_for_device_manager == nullptr) {
        return InternalError("Host callback not supported for runtime type: %s",
                             client->runtime_type());
      }
      returned_futures.emplace();

      host_callback_states = std::make_shared<HostCallbackStates>();

      for (int i = 0; i < num_computations; ++i) {
        auto& contexts = host_callback_states->contexts.emplace_back();
        auto& send_callbacks =
            host_callback_states->send_callbacks.emplace_back();
        auto& recv_callbacks =
            host_callback_states->recv_callbacks.emplace_back();

        for (const py::capsule& host_callback : host_callbacks) {
          contexts.push_back(CreateHostCallbackStateAndAppendSendRecvCallbacks(
              *host_callback.get_pointer<HostCallback>(),
              host_memory_for_device_manager, send_callbacks, recv_callbacks));
        }
      }
      opts.send_callbacks = host_callback_states->send_callbacks;
      opts.recv_callbacks = host_callback_states->recv_callbacks;
    }

    py::gil_scoped_release gil_release;
    for (const auto& arg : args) {
      if (ArgAdapter::num_devices(arg) != num_computations) {
        return xla::InvalidArgument(
            "Expected args to execute_sharded_on_local_devices to have %d "
            "shards, got: [%s]",
            num_computations,
            absl::StrJoin(args, ", ", [](std::string* out, const ArgT& arg) {
              out->append(std::to_string(ArgAdapter::num_devices(arg)));
            }));
      }
    }
#ifdef JAX_ENABLE_IFRT
    std::vector<ifrt::Array*> arg_arrays(args.size());
    std::vector<std::unique_ptr<ifrt::Array>> owned_ifrt_arrays;
    owned_ifrt_arrays.reserve(args.size());
    absl::c_transform(args, arg_arrays.begin(), [&](const ArgT& arg) mutable {
      return ArgAdapter::GetIfRtArray(arg, owned_ifrt_arrays);
    });
#else
    std::vector<std::vector<PjRtBuffer*>> arg_buffers(num_computations);
    const int num_args = args.size();
    for (int computation = 0; computation < num_computations; ++computation) {
      arg_buffers[computation].resize(num_args);
      absl::c_transform(args, arg_buffers[computation].begin(),
                        [&](const ArgT& arg) {
                          return ArgAdapter::GetPjRtBuffer(arg, computation);
                        });
    }
#endif
#ifdef JAX_ENABLE_IFRT
    TF_ASSIGN_OR_RETURN(
        auto result, ifrt_loaded_executable->Execute(arg_arrays, opts,
                                                     /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
    if (returned_futures.has_value()) {
      returned_futures->resize(num_computations, std::move(result.status));
    }
#else
    TF_ASSIGN_OR_RETURN(output_buffers, executable->Execute(arg_buffers, opts,
                                                            returned_futures));
#endif

    if (!host_callbacks.empty()) {
      // For host callbacks to work, `returned_futures` must not be nullopt.
#ifdef JAX_ENABLE_IFRT
      returned_futures.value().at(0).OnReady(
          [host_callback_states](Status) mutable {
            host_callback_states.reset();
          });
#else
      for (int i = 0; i < num_computations; ++i) {
        returned_futures.value().at(i).OnReady(
            [host_callback_states](Status) mutable {
              host_callback_states.reset();
            });
      }
#endif
    }
  }

  std::vector<ResultT> outputs;
#ifdef JAX_ENABLE_IFRT
  PopulateExecuteShardedResults(client, std::move(output_arrays),
                                num_computations, outputs);
#else
  PopulateExecuteShardedResults(client, std::move(output_buffers), outputs);
#endif

  // TODO(b/240696624): Although the PjRt interface require `returned_futures`
  // to be resized correctly if it is not nullopt, some implementation does not
  // implement this. So we have to check whether returned_futures is empty.
  // Remove this check once the implementation is fixed.
  if (!returned_futures.has_value()) {
    return std::pair<std::vector<ResultT>, PyShardedToken>(std::move(outputs),
                                                           PyShardedToken());
  }

  PyShardedToken py_sharded_token(std::move(*returned_futures));
  return std::pair<std::vector<ResultT>, PyShardedToken>(
      std::move(outputs), std::move(py_sharded_token));
}

}  // namespace

StatusOr<std::vector<PyShardedBuffer>>
PyLoadedExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<PyShardedBuffer* const> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
#ifdef JAX_ENABLE_IFRT
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, ifrt_loaded_executable_.get(),
                          host_callbacks_, args, returned_futures));
#else
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, executable_.get(), host_callbacks_,
                          args, returned_futures));
#endif
  return std::move(outputs_and_tokens.first);
}

StatusOr<std::pair<std::vector<PyShardedBuffer>, PyShardedToken>>
PyLoadedExecutable::ExecuteShardedOnLocalDevicesWithTokens(
    absl::Span<PyShardedBuffer* const> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
#ifdef JAX_ENABLE_IFRT
  returned_futures.emplace();
  return ExecuteShardedOnLocalDevicesInternal(
      options_, client_, ifrt_loaded_executable_.get(), host_callbacks_, args,
      returned_futures);
#else
  if (executable_->IsReturnedFutureSupported()) returned_futures.emplace();
  return ExecuteShardedOnLocalDevicesInternal(
      options_, client_, executable_.get(), host_callbacks_, args,
      returned_futures);
#endif
}

StatusOr<std::vector<std::vector<PyBuffer::object>>>
PyLoadedExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<const std::vector<PyBuffer::object>> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
#ifdef JAX_ENABLE_IFRT
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, ifrt_loaded_executable_.get(),
                          host_callbacks_, args, returned_futures));
#else
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, executable_.get(), host_callbacks_,
                          args, returned_futures));
#endif
  return std::move(outputs_and_tokens.first);
}

StatusOr<std::pair<std::vector<std::vector<PyBuffer::object>>, PyShardedToken>>
PyLoadedExecutable::ExecuteShardedOnLocalDevicesWithTokens(
    absl::Span<const std::vector<PyBuffer::object>> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
#ifdef JAX_ENABLE_IFRT
  returned_futures.emplace();
  return ExecuteShardedOnLocalDevicesInternal(
      options_, client_, ifrt_loaded_executable_.get(), host_callbacks_, args,
      returned_futures);
#else

  if (executable_->IsReturnedFutureSupported()) returned_futures.emplace();
  return ExecuteShardedOnLocalDevicesInternal(
      options_, client_, executable_.get(), host_callbacks_, args,
      returned_futures);
#endif
}

StatusOr<std::vector<std::shared_ptr<HloModule>>>
PyLoadedExecutable::HloModules() const {
#ifdef JAX_ENABLE_IFRT
  return ifrt_loaded_executable_->GetHloModules();
#else
  return executable_->GetHloModules();
#endif
}

std::optional<std::vector<OpSharding>>
PyLoadedExecutable::GetParameterShardings() const {
#ifdef JAX_ENABLE_IFRT
  return ifrt_loaded_executable_->GetParameterShardings();
#else
  return executable_->GetParameterShardings();
#endif
}

std::optional<std::vector<OpSharding>> PyLoadedExecutable::GetOutputShardings()
    const {
#ifdef JAX_ENABLE_IFRT
  return ifrt_loaded_executable_->GetOutputShardings();
#else
  return executable_->GetOutputShardings();
#endif
}

void PyLoadedExecutable::KeepAlive(py::object obj) {
  keepalives_.push_back(std::move(obj));
}

}  // namespace xla
