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
#include "tensorflow/compiler/xla/python/host_callback.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace xla {

namespace py = pybind11;

PyExecutable::PyExecutable(std::shared_ptr<PyClient> client,
                           std::unique_ptr<PjRtExecutable> executable,
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
    options_.launch_id = tensorflow::Fingerprint32(*fingerprint_);
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

StatusOr<std::vector<PyBuffer::object>> PyExecutable::Execute(
    absl::Span<PyBuffer::object const> args) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  {
    auto options = options_;
    std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
    std::shared_ptr<HostCallbackStates> host_callback_states;
    if (!host_callbacks_.empty()) {
      returned_futures.emplace();

      host_callback_states = std::make_shared<HostCallbackStates>();
      auto& contexts = host_callback_states->contexts.emplace_back();
      auto& send_callbacks =
          host_callback_states->send_callbacks.emplace_back();
      auto& recv_callbacks =
          host_callback_states->recv_callbacks.emplace_back();

      for (const py::capsule& host_callback : host_callbacks_) {
        contexts.push_back(CreateHostCallbackStateAndAppendSendRecvCallbacks(
            host_callback.get_pointer<HostCallback>(), client()->pjrt_client(),
            send_callbacks, recv_callbacks));
      }
      options.send_callbacks = host_callback_states->send_callbacks;
      options.recv_callbacks = host_callback_states->recv_callbacks;
    }

    py::gil_scoped_release gil_release;
    std::vector<PjRtBuffer*> arg_buffers(args.size());
    absl::c_transform(
        args, arg_buffers.begin(),
        [](const PyBuffer::object& buf) { return buf.buf()->buffer(); });
    TF_ASSIGN_OR_RETURN(
        output_buffers,
        executable_->Execute({arg_buffers}, options, returned_futures));

    if (!host_callbacks_.empty()) {
      returned_futures.value().at(0).OnReady(
          [host_callback_states](Status) mutable {
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
  return outputs;
}

StatusOr<std::vector<std::vector<PyBuffer::object>>>
PyExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<const std::vector<PyBuffer::object>> args) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  int num_computations = executable_->addressable_devices().size();
  {
    py::gil_scoped_release gil_release;
    for (const auto& arg : args) {
      if (arg.size() != num_computations) {
        return xla::InvalidArgument(
            "Expected args to execute_sharded_on_local_devices to have %d "
            "shards, got: [%s]",
            num_computations,
            absl::StrJoin(
                args, ", ",
                [](std::string* out, const std::vector<PyBuffer::object>& arg) {
                  out->append(std::to_string(arg.size()));
                }));
      }
    }
    std::vector<std::vector<PjRtBuffer*>> arg_buffers(num_computations);
    const int num_args = args.size();
    for (int computation = 0; computation < num_computations; ++computation) {
      arg_buffers[computation].resize(num_args);
      absl::c_transform(args, arg_buffers[computation].begin(),
                        [&](const std::vector<PyBuffer::object>& arg) {
                          return arg[computation].buf()->buffer();
                        });
    }
    TF_ASSIGN_OR_RETURN(output_buffers,
                        executable_->Execute(arg_buffers, options_));
  }
  auto traceback = Traceback::Get();
  int num_output_buffers = output_buffers[0].size();
  std::vector<std::vector<PyBuffer::object>> outputs;
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    outputs[buffer_id].reserve(num_computations);
    for (int computation = 0; computation < num_computations; ++computation) {
      outputs[buffer_id].push_back(PyBuffer::Make(
          client_, std::move(output_buffers[computation][buffer_id]),
          traceback));
    }
  }
  return outputs;
}

StatusOr<std::vector<std::shared_ptr<HloModule>>> PyExecutable::HloModules()
    const {
  return executable_->GetHloModules();
}

void PyExecutable::KeepAlive(py::object obj) {
  keepalives_.push_back(std::move(obj));
}

}  // namespace xla
