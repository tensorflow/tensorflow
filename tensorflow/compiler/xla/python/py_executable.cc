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

#include <string>
#include <utility>

#include "absl/algorithm/container.h"
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

PyExecutable::PyExecutable(std::shared_ptr<PyClient> client,
                           std::unique_ptr<PjRtLoadedExecutable> executable,
                           std::shared_ptr<Traceback> traceback,
                           std::optional<std::string> fingerprint,
                           std::vector<pybind11::capsule> host_callbacks)
    : client_(std::move(client)),
      executable_(std::move(executable)),
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
    VLOG(1) << "Fingerprint for executable " << executable_->name() << ": "
            << *fingerprint_;
  }
}

PyExecutable::~PyExecutable() {
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

std::vector<ClientAndPtr<PjRtDevice>> PyExecutable::AddressableDevices() const {
  std::vector<ClientAndPtr<PjRtDevice>> devices;
  devices.reserve(executable_->addressable_devices().size());
  for (PjRtDevice* device : executable_->addressable_devices()) {
    devices.push_back(WrapWithClient(client_, device));
  }
  return devices;
}

StatusOr<std::pair<std::vector<PyBuffer::object>, PyToken>>
PyExecutable::ExecuteInternal(
    absl::Span<PyBuffer::object const> args, PjRtDevice* device,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
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
    std::vector<PjRtBuffer*> arg_buffers(args.size());
    absl::c_transform(
        args, arg_buffers.begin(),
        [](const PyBuffer::object& buf) { return buf.buf()->buffer(); });
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

    if (!host_callbacks_.empty()) {
      // For host callbacks to work, `returned_futures` must not be nullopt.
      returned_futures->at(0).OnReady([host_callback_states](Status) mutable {
        host_callback_states.reset();
      });
    }
  }
  auto traceback = Traceback::Get();
  std::vector<PyBuffer::object> outputs;
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
}

StatusOr<std::pair<std::vector<PyBuffer::object>, PyToken>>
PyExecutable::ExecuteWithToken(absl::Span<PyBuffer::object const> args,
                               PjRtDevice* device) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  if (executable_->IsReturnedFutureSupported()) returned_futures.emplace();
  return ExecuteInternal(args, device, returned_futures);
}

StatusOr<std::vector<PyBuffer::object>> PyExecutable::Execute(
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
    return arg->num_devices();
  }
  static PjRtBuffer* GetPjRtBuffer(const PyShardedBuffer* arg, int device_id) {
    return arg->GetPjRtBuffer(device_id);
  }
};

template <>
struct ShardedBufferAdapter<std::vector<PyBuffer::object>> {
  using ResultT = std::vector<PyBuffer::object>;
  static int num_devices(const std::vector<PyBuffer::object>& arg) {
    return arg.size();
  }
  static PjRtBuffer* GetPjRtBuffer(const std::vector<PyBuffer::object>& arg,
                                   int device_id) {
    return arg.at(device_id).buf()->buffer();
  }
};

void PopulateExecuteShardedResults(
    const std::shared_ptr<PyClient>& client,
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> pjrt_buffers,
    std::vector<PyShardedBuffer>& outputs) {
  auto traceback = Traceback::Get();
  int num_computations = pjrt_buffers.size();
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = pjrt_buffers[0].size();
  outputs.reserve(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    std::vector<std::shared_ptr<PjRtBuffer>> buffers;
    buffers.reserve(num_computations);
    for (int computation = 0; computation < num_computations; ++computation) {
      buffers.push_back(std::move(pjrt_buffers[computation][buffer_id]));
    }
    outputs.emplace_back(client, std::move(buffers), traceback);
  }
}

void PopulateExecuteShardedResults(
    const std::shared_ptr<PyClient>& client,
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> pjrt_buffers,
    std::vector<std::vector<PyBuffer::object>>& outputs) {
  auto traceback = Traceback::Get();
  int num_computations = pjrt_buffers.size();
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = pjrt_buffers[0].size();
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    outputs[buffer_id].reserve(num_computations);
    for (int computation = 0; computation < num_computations; ++computation) {
      outputs[buffer_id].push_back(PyBuffer::Make(
          client, std::move(pjrt_buffers[computation][buffer_id]), traceback));
    }
  }
}

template <typename ArgT,
          typename ResultT = typename ShardedBufferAdapter<ArgT>::ResultT,
          typename ArgAdapter = ShardedBufferAdapter<ArgT>>
StatusOr<std::pair<std::vector<ResultT>, PyShardedToken>>
ExecuteShardedOnLocalDevicesInternal(
    const ExecuteOptions& options, const std::shared_ptr<PyClient>& client,
    PjRtLoadedExecutable* executable,
    absl::Span<const py::capsule> host_callbacks, absl::Span<const ArgT> args,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  int num_computations = executable->addressable_devices().size();
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
    std::vector<std::vector<PjRtBuffer*>> arg_buffers(num_computations);
    const int num_args = args.size();
    for (int computation = 0; computation < num_computations; ++computation) {
      arg_buffers[computation].resize(num_args);
      absl::c_transform(args, arg_buffers[computation].begin(),
                        [&](const ArgT& arg) {
                          return ArgAdapter::GetPjRtBuffer(arg, computation);
                        });
    }
    TF_ASSIGN_OR_RETURN(output_buffers, executable->Execute(arg_buffers, opts,
                                                            returned_futures));

    if (!host_callbacks.empty()) {
      // For host callbacks to work, `returned_futures` must not be nullopt.
      for (int i = 0; i < num_computations; ++i) {
        returned_futures.value().at(i).OnReady(
            [host_callback_states](Status) mutable {
              host_callback_states.reset();
            });
      }
    }
  }

  std::vector<ResultT> outputs;
  PopulateExecuteShardedResults(client, std::move(output_buffers), outputs);

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
PyExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<PyShardedBuffer* const> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, executable_.get(), host_callbacks_,
                          args, returned_futures));
  return std::move(outputs_and_tokens.first);
}

StatusOr<std::pair<std::vector<PyShardedBuffer>, PyShardedToken>>
PyExecutable::ExecuteShardedOnLocalDevicesWithTokens(
    absl::Span<PyShardedBuffer* const> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  if (executable_->IsReturnedFutureSupported()) returned_futures.emplace();
  return ExecuteShardedOnLocalDevicesInternal(
      options_, client_, executable_.get(), host_callbacks_, args,
      returned_futures);
}

StatusOr<std::vector<std::vector<PyBuffer::object>>>
PyExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<const std::vector<PyBuffer::object>> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, executable_.get(), host_callbacks_,
                          args, returned_futures));
  return std::move(outputs_and_tokens.first);
}

StatusOr<std::pair<std::vector<std::vector<PyBuffer::object>>, PyShardedToken>>
PyExecutable::ExecuteShardedOnLocalDevicesWithTokens(
    absl::Span<const std::vector<PyBuffer::object>> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  if (executable_->IsReturnedFutureSupported()) returned_futures.emplace();
  return ExecuteShardedOnLocalDevicesInternal(
      options_, client_, executable_.get(), host_callbacks_, args,
      returned_futures);
}

StatusOr<std::vector<std::shared_ptr<HloModule>>> PyExecutable::HloModules()
    const {
  return executable_->GetHloModules();
}

void PyExecutable::KeepAlive(py::object obj) {
  keepalives_.push_back(std::move(obj));
}

}  // namespace xla
