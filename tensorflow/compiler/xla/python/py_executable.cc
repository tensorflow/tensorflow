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
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/pjrt/host_callback.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/executable.h"
#include "tensorflow/compiler/xla/python/ifrt/future.h"
#include "tensorflow/tsl/platform/fingerprint.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

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
    std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable,
    std::shared_ptr<Traceback> traceback,
    std::optional<std::string> fingerprint,
    std::vector<pybind11::capsule> host_callbacks)
    : client_(std::move(client)),
      ifrt_loaded_executable_(std::move(ifrt_loaded_executable)),
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
    VLOG(1) << "Fingerprint for executable " << ifrt_loaded_executable_->name()
            << ": " << *fingerprint_;
  }
  options_.use_major_to_minor_data_layout_for_callbacks = true;
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
  devices.reserve(ifrt_loaded_executable_->addressable_devices().size());
  for (ifrt::Device* device : ifrt_loaded_executable_->addressable_devices()) {
    devices.push_back(WrapWithClient(client_, device));
  }
  return devices;
}

namespace {

// Traits classes of common methods for std::vector<PyArray>.
template <typename ShardedBufferT>
struct ShardedBufferAdapter;

template <>
struct ShardedBufferAdapter<ExecuteShardedArg> {
  static int num_devices(const ExecuteShardedArg& arg) {
    if (std::holds_alternative<PyArray>(arg)) {
      CHECK(std::get<PyArray>(arg).fastpath_enabled());
      return std::get<PyArray>(arg).num_addressable_shards();
    } else {
      return std::get<std::vector<PyArray>>(arg).size();
    }
  }
  static tsl::RCReference<ifrt::Array> GetIfRtArray(
      const ExecuteShardedArg& arg) {
    if (std::holds_alternative<PyArray>(arg)) {
      CHECK(std::get<PyArray>(arg).fastpath_enabled());
      return tsl::FormRef(std::get<PyArray>(arg).ifrt_array());
    }
    auto& arg_vector = std::get<std::vector<PyArray>>(arg);

    // TODO(hyeontaek): This on-demand Array creation is not efficient and has
    // insufficient information about the shape (a dummy shape is used). This
    // should be removed if possible and only be used in the context where the
    // shape information is unused.
    std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
    ifrt_arrays.reserve(arg_vector.size());
    ifrt::DeviceList::Devices devices;
    devices.reserve(arg_vector.size());
    for (auto& arr : arg_vector) {
      CHECK_EQ(arr.ifrt_array()->sharding().devices().size(), 1)
          << arr.ifrt_array()->sharding().DebugString();
      ifrt_arrays.push_back(tsl::FormRef(arr.ifrt_array()));
      devices.push_back(arr.ifrt_array()->sharding().devices().front());
    }
    CHECK(!ifrt_arrays.empty());
    // Use a dummy shape.
    // TODO(hyeontaek): Find a way to compute a correct shape.
    auto ifrt_array =
        ifrt_arrays.front()->client()->AssembleArrayFromSingleDeviceArrays(
            ifrt_arrays.front()->shape(),
            ifrt::OpaqueSharding::Create(ifrt::DeviceList(std::move(devices))),
            absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput);
    TF_CHECK_OK(ifrt_array.status());
    return *ifrt_array;
  }
};

void PopulateExecuteShardedResults(
    const std::shared_ptr<PyClient>& client,
    std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays,
    int num_computations, std::vector<std::vector<PyArray>>& outputs) {
  auto traceback = Traceback::Get();
  DCHECK_GT(num_computations, 0);
  int num_output_buffers = ifrt_arrays.size();
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    outputs[buffer_id].reserve(num_computations);
    auto exploded_arrays =
        ifrt_arrays[buffer_id]->DisassembleIntoSingleDeviceArrays(
            ifrt::ArrayCopySemantics::kReuseInput);
    TF_CHECK_OK(exploded_arrays.status());
    for (auto& exploded_array : *exploded_arrays) {
      outputs[buffer_id].push_back(PyArray::MakeFromSingleDeviceArray(
          client, traceback, std::move(exploded_array), false, true));
    }
  }
}

template <typename ArgT, typename ArgAdapter = ShardedBufferAdapter<ArgT>>
StatusOr<PyExecuteResults> ExecuteShardedOnLocalDevicesInternal(
    const ExecuteOptions& options, const std::shared_ptr<PyClient>& client,
    ifrt::LoadedExecutable* ifrt_loaded_executable,
    absl::Span<const py::capsule> host_callbacks, absl::Span<const ArgT> args,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  std::vector<tsl::RCReference<ifrt::Array>> output_arrays;
  std::unique_ptr<ifrt::Future<Status>> returned_future;
  int num_computations = ifrt_loaded_executable->addressable_devices().size();
  {
    auto opts = options;
    std::shared_ptr<HostCallbackStates> host_callback_states;
    if (!host_callbacks.empty()) {
      if (!client->pjrt_client()->SupportsSendRecvCallbacks()) {
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
              /*host_memory_for_device_manager=*/nullptr, send_callbacks,
              recv_callbacks,
              /*use_major_to_minor_data_layout_for_callbacks=*/true));
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
    std::vector<tsl::RCReference<ifrt::Array>> arg_arrays(args.size());
    absl::c_transform(args, arg_arrays.begin(), [&](const ArgT& arg) mutable {
      return ArgAdapter::GetIfRtArray(arg);
    });
    TF_ASSIGN_OR_RETURN(auto result, ifrt_loaded_executable->Execute(
                                         absl::MakeSpan(arg_arrays), opts,
                                         /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
    if (returned_futures.has_value()) {
      returned_futures->resize(num_computations, std::move(result.status));
    }

    if (!host_callbacks.empty()) {
      // For host callbacks to work, `returned_futures` must not be nullopt.
      returned_futures.value().at(0).OnReady(
          [host_callback_states](Status) mutable {
            host_callback_states.reset();
          });
    }
  }

  // TODO(b/240696624): Although the PjRt interface require `returned_futures`
  // to be resized correctly if it is not nullopt, some implementation does not
  // implement this. So we have to check whether returned_futures is empty.
  // Remove this check once the implementation is fixed.
  auto py_sharded_token = returned_futures.has_value()
                              ? PyShardedToken(std::move(*returned_futures))
                              : PyShardedToken();

  return PyExecuteResults(client, std::move(output_arrays), num_computations,
                          std::move(py_sharded_token));
}

}  // namespace

PyExecuteResults::PyExecuteResults(
    const std::shared_ptr<PyClient>& client,
    std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays,
    int num_computations, PyShardedToken token)
    : client_(client),
      ifrt_arrays_(std::move(ifrt_arrays)),
      num_computations_(num_computations),
      token_(std::move(token)) {}

void PyExecuteResults::CheckNotDisassembled() const {
  if (is_exploded_) {
    throw py::value_error("ExecuteResults already exploded.");
  }
}

std::vector<tsl::RCReference<ifrt::Array>> PyExecuteResults::Consume() {
  CheckNotDisassembled();
  is_exploded_ = true;
  return std::move(ifrt_arrays_);
}

PyShardedToken PyExecuteResults::ConsumeToken() {
  if (token_consumed_) {
    throw py::value_error("ExecuteResults token already consumed.");
  }
  token_consumed_ = true;
  return std::move(token_);
}

std::vector<std::vector<PyArray>>
PyExecuteResults::DisassembleIntoSingleDeviceArrays() {
  std::vector<std::vector<PyArray>> outputs;
  PopulateExecuteShardedResults(client_, Consume(), num_computations_, outputs);
  return outputs;
}

std::vector<std::vector<PyArray>>
PyExecuteResults::DisassemblePrefixIntoSingleDeviceArrays(size_t n) {
  CheckNotDisassembled();
  if (n > ifrt_arrays_.size()) {
    throw py::value_error(
        absl::StrCat("In DisassemblePrefixIntoSingleDeviceArrays: ", n, " > ",
                     ifrt_arrays_.size()));
  }
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  ifrt_arrays.reserve(ifrt_arrays_.size() - n);
  for (size_t i = n; i < ifrt_arrays_.size(); ++i) {
    ifrt_arrays.push_back(std::move(ifrt_arrays_[i]));
  }
  ifrt_arrays_.erase(ifrt_arrays_.begin() + n, ifrt_arrays_.end());
  std::swap(ifrt_arrays_, ifrt_arrays);
  std::vector<std::vector<PyArray>> outputs;
  PopulateExecuteShardedResults(client_, std::move(ifrt_arrays),
                                num_computations_, outputs);
  return outputs;
}

std::vector<py::object> PyExecuteResults::ConsumeWithHandlers(
    std::vector<std::variant<const PyArrayResultHandler*, py::object>>
        out_handlers) {
  std::vector<py::object> outputs;
  auto ifrt_arrays = Consume();
  auto traceback = Traceback::Get();
  DCHECK_GT(num_computations_, 0);
  int num_output_buffers = ifrt_arrays.size();
  outputs.reserve(num_output_buffers);
  if (out_handlers.size() != num_output_buffers) {
    throw py::value_error(absl::StrCat(
        "Mismatch between out_handlers and num_results: ", out_handlers.size(),
        " vs ", num_output_buffers));
  }
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    auto& handler = out_handlers[buffer_id];
    if (std::holds_alternative<const PyArrayResultHandler*>(handler)) {
      outputs.push_back(std::get<const PyArrayResultHandler*>(handler)->Call(
          client_, std::move(ifrt_arrays[buffer_id])));
    } else {
      tsl::profiler::TraceMe traceme("ConsumeWithHandlers fallback.");
      std::vector<PyArray> bufs;
      bufs.reserve(num_computations_);
      auto disassembled_arrays =
          ifrt_arrays[buffer_id]->DisassembleIntoSingleDeviceArrays(
              ifrt::ArrayCopySemantics::kReuseInput);
      TF_CHECK_OK(disassembled_arrays.status());
      for (auto& disassembled_array : *disassembled_arrays) {
        bufs.push_back(PyArray::MakeFromSingleDeviceArray(
            client_, traceback, std::move(disassembled_array), false, true));
      }
      outputs.push_back(std::get<py::object>(handler)(std::move(bufs)));
    }
  }
  return outputs;
}

StatusOr<std::vector<std::vector<PyArray>>>
PyLoadedExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<const ExecuteShardedArg> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, ifrt_loaded_executable_.get(),
                          host_callbacks_, args, returned_futures));
  return outputs_and_tokens.DisassembleIntoSingleDeviceArrays();
}

StatusOr<std::pair<std::vector<std::vector<PyArray>>, PyShardedToken>>
PyLoadedExecutable::ExecuteShardedOnLocalDevicesWithTokens(
    absl::Span<const ExecuteShardedArg> args) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  returned_futures.emplace();
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options_, client_, ifrt_loaded_executable_.get(),
                          host_callbacks_, args, returned_futures));
  return std::make_pair(outputs_and_tokens.DisassembleIntoSingleDeviceArrays(),
                        outputs_and_tokens.ConsumeToken());
}

StatusOr<PyExecuteResults> PyLoadedExecutable::ExecuteSharded(
    std::vector<ExecuteShardedArg> args, bool with_tokens) {
  std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
  if (with_tokens) {
    returned_futures.emplace();
  }
  absl::Span<const ExecuteShardedArg> span_args = args;
  return ExecuteShardedOnLocalDevicesInternal(
      options_, client_, ifrt_loaded_executable_.get(), host_callbacks_,
      span_args, returned_futures);
}

StatusOr<std::vector<std::shared_ptr<HloModule>>>
PyLoadedExecutable::HloModules() const {
  return ifrt_loaded_executable_->GetHloModules();
}

std::optional<std::vector<OpSharding>>
PyLoadedExecutable::GetParameterShardings() const {
  return ifrt_loaded_executable_->GetParameterShardings();
}

std::optional<std::vector<OpSharding>> PyLoadedExecutable::GetOutputShardings()
    const {
  return ifrt_loaded_executable_->GetOutputShardings();
}

void PyLoadedExecutable::KeepAlive(py::object obj) {
  keepalives_.push_back(std::move(obj));
}

}  // namespace xla
