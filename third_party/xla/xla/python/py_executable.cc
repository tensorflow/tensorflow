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

#include "xla/python/py_executable.h"

#include <Python.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/py_array.h"
#include "xla/python/py_client.h"
#include "xla/python/py_device.h"
#include "xla/python/traceback.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

namespace nb = nanobind;

absl::Status PyToken::Await() {
  CHECK(future_.IsValid());
  nb::gil_scoped_release gil_release;
  return future_.Await();
}

absl::Status PyShardedToken::Await() {
  nb::gil_scoped_release gil_release;
  absl::Status status = absl::OkStatus();
  for (auto& future : futures_) {
    auto s = future.Await();
    if (!s.ok()) status = std::move(s);
  }
  return status;
}

PyLoadedExecutable::PyLoadedExecutable(
    nb_class_ptr<PyClient> client,
    std::shared_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable,
    std::optional<nb_traceback> traceback,
    std::optional<std::string> fingerprint)
    : client_(std::move(client)),
      ifrt_loaded_executable_(std::move(ifrt_loaded_executable)),
      traceback_(std::move(traceback)),
      fingerprint_(std::move(fingerprint)) {
  CHECK(PyGILState_Check());
  if (fingerprint_) {
    options_.launch_id = tsl::Fingerprint32(*fingerprint_);
    VLOG(1) << "Fingerprint for executable " << ifrt_loaded_executable_->name()
            << ": " << *fingerprint_;
  }
  nb::ft_lock_guard lock(client_->executables_mutex_);
  next_ = client_->executables_;
  client_->executables_ = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
}

PyLoadedExecutable::~PyLoadedExecutable() {
  CHECK(PyGILState_Check());
  nb::ft_lock_guard lock(client_->executables_mutex_);
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

std::vector<nb_class_ptr<PyDevice>> PyLoadedExecutable::AddressableDevices()
    const {
  std::vector<nb_class_ptr<PyDevice>> devices;
  devices.reserve(ifrt_loaded_executable_->addressable_devices().size());
  for (ifrt::Device* device : ifrt_loaded_executable_->addressable_devices()) {
    devices.push_back(client_->GetPyDevice(device));
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
      return std::get<PyArray>(arg).num_addressable_shards();
    } else {
      return std::get<std::vector<PyArray>>(arg).size();
    }
  }
  static tsl::RCReference<ifrt::Array> GetIfRtArray(
      const ExecuteShardedArg& arg) {
    if (std::holds_alternative<PyArray>(arg)) {
      return tsl::FormRef(std::get<PyArray>(arg).ifrt_array());
    }
    auto& arg_vector = std::get<std::vector<PyArray>>(arg);

    // TODO(hyeontaek): This on-demand Array creation is not efficient and has
    // insufficient information about the shape (a dummy shape is used). This
    // should be removed if possible and only be used in the context where the
    // shape information is unused.
    std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
    ifrt_arrays.reserve(arg_vector.size());
    ifrt::BasicDeviceList::Devices devices;
    devices.reserve(arg_vector.size());
    for (auto& arr : arg_vector) {
      CHECK_EQ(arr.ifrt_array()->sharding().devices()->size(), 1)
          << arr.ifrt_array()->sharding().DebugString();
      ifrt_arrays.push_back(tsl::FormRef(arr.ifrt_array()));
      devices.push_back(
          arr.ifrt_array()->sharding().devices()->devices().front());
    }
    CHECK(!ifrt_arrays.empty());
    // Use a dummy shape.
    // TODO(hyeontaek): Find a way to compute a correct shape.
    // TODO(yashkatariya): Plumb sharding or memory_kind here.
    auto ifrt_array =
        ifrt_arrays.front()->client()->AssembleArrayFromSingleDeviceArrays(
            ifrt_arrays.front()->shape(),
            ifrt::OpaqueSharding::Create(
                ifrt::BasicDeviceList::Create(std::move(devices)),
                ifrt::MemoryKind()),
            absl::MakeSpan(ifrt_arrays), ifrt::ArrayCopySemantics::kReuseInput);
    TF_CHECK_OK(ifrt_array.status());
    return *ifrt_array;
  }
};

void PopulateExecuteShardedResults(
    const nb_class_ptr<PyClient>& client,
    std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays,
    const PjRtFuture<>& result_status, int num_computations,
    std::vector<std::vector<PyArray>>& outputs) {
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
          client, traceback, std::move(exploded_array), false, true,
          result_status));
    }
  }
}

template <typename ArgT, typename ArgAdapter = ShardedBufferAdapter<ArgT>>
absl::StatusOr<PyExecuteResults> ExecuteShardedOnLocalDevicesInternal(
    const ifrt::ExecuteOptions& options, const nb_class_ptr<PyClient>& client,
    ifrt::LoadedExecutable* ifrt_loaded_executable, absl::Span<const ArgT> args,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  std::vector<tsl::RCReference<ifrt::Array>> output_arrays;
  std::unique_ptr<ifrt::Future<>> returned_future;
  int num_computations = ifrt_loaded_executable->addressable_devices().size();
  PjRtFuture<> result_status;
  {
    nb::gil_scoped_release gil_release;
    for (const auto& arg : args) {
      if (ArgAdapter::num_devices(arg) != num_computations) {
        return InvalidArgument(
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
                                         absl::MakeSpan(arg_arrays), options,
                                         /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
    // options.fill_status is only supposed to be true when the computation has
    // tokens.
    if (options.fill_status) {
      result_status = result.status;
      if (returned_futures.has_value()) {
        returned_futures->resize(num_computations, std::move(result.status));
      }
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
                          std::move(py_sharded_token), result_status);
}

}  // namespace

PyExecuteResults::PyExecuteResults(
    const nb_class_ptr<PyClient>& client,
    std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays,
    int num_computations, PyShardedToken token, PjRtFuture<> result_status)
    : client_(client),
      ifrt_arrays_(std::move(ifrt_arrays)),
      num_computations_(num_computations),
      token_(std::move(token)),
      result_status_(std::move(result_status)) {}

void PyExecuteResults::CheckNotDisassembled() const {
  if (is_exploded_) {
    throw nb::value_error("ExecuteResults already exploded.");
  }
}

std::vector<tsl::RCReference<ifrt::Array>> PyExecuteResults::Consume() {
  CheckNotDisassembled();
  is_exploded_ = true;
  return std::move(ifrt_arrays_);
}

PyShardedToken PyExecuteResults::ConsumeToken() {
  if (token_consumed_) {
    throw nb::value_error("ExecuteResults token already consumed.");
  }
  token_consumed_ = true;
  return std::move(token_);
}

std::vector<std::vector<PyArray>>
PyExecuteResults::DisassembleIntoSingleDeviceArrays() {
  std::vector<std::vector<PyArray>> outputs;
  PopulateExecuteShardedResults(
      client_, Consume(),
      result_status_.IsValid() ? result_status_ : PjRtFuture<>(),
      num_computations_, outputs);
  return outputs;
}

std::vector<std::vector<PyArray>>
PyExecuteResults::DisassemblePrefixIntoSingleDeviceArrays(size_t n) {
  CheckNotDisassembled();
  if (n > ifrt_arrays_.size()) {
    throw nb::value_error(
        absl::StrCat("In DisassemblePrefixIntoSingleDeviceArrays: ", n, " > ",
                     ifrt_arrays_.size())
            .c_str());
  }
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays;
  ifrt_arrays.reserve(ifrt_arrays_.size() - n);
  for (size_t i = n; i < ifrt_arrays_.size(); ++i) {
    ifrt_arrays.push_back(std::move(ifrt_arrays_[i]));
  }
  ifrt_arrays_.erase(ifrt_arrays_.begin() + n, ifrt_arrays_.end());
  std::swap(ifrt_arrays_, ifrt_arrays);
  std::vector<std::vector<PyArray>> outputs;
  PopulateExecuteShardedResults(
      client_, std::move(ifrt_arrays),
      result_status_.IsValid() ? result_status_ : PjRtFuture<>(),
      num_computations_, outputs);
  return outputs;
}

std::vector<nb::object> PyExecuteResults::ConsumeWithHandlers(
    std::vector<std::variant<const PyArrayResultHandler*, nb::object>>
        out_handlers) {
  std::vector<nb::object> outputs;
  auto ifrt_arrays = Consume();
  auto traceback = Traceback::Get();
  DCHECK_GT(num_computations_, 0);
  int num_output_buffers = ifrt_arrays.size();
  outputs.reserve(num_output_buffers);
  if (out_handlers.size() != num_output_buffers) {
    throw nb::value_error(
        absl::StrCat("Mismatch between out_handlers and num_results: ",
                     out_handlers.size(), " vs ", num_output_buffers)
            .c_str());
  }
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    auto& handler = out_handlers[buffer_id];
    if (std::holds_alternative<const PyArrayResultHandler*>(handler)) {
      outputs.push_back(std::get<const PyArrayResultHandler*>(handler)->Call(
          client_, std::move(ifrt_arrays[buffer_id]),
          result_status_.IsValid() ? result_status_ : PjRtFuture<>()));
    } else {
      tsl::profiler::TraceMe traceme("ConsumeWithHandlers fallback.");
      auto disassembled_arrays =
          ifrt_arrays[buffer_id]->DisassembleIntoSingleDeviceArrays(
              ifrt::ArrayCopySemantics::kReuseInput);
      TF_CHECK_OK(disassembled_arrays.status());
      nb::list bufs =
          nb::steal<nb::list>(PyList_New(disassembled_arrays->size()));
      int i = 0;
      for (auto& disassembled_array : *disassembled_arrays) {
        nb::object array = PyArray::MakeFromSingleDeviceArray(
            client_, traceback, std::move(disassembled_array), false, true,
            result_status_.IsValid() ? result_status_ : PjRtFuture<>());
        PyList_SET_ITEM(bufs.ptr(), i, array.release().ptr());
        ++i;
      }
      outputs.push_back(std::get<nb::object>(handler)(std::move(bufs)));
    }
  }
  return outputs;
}

absl::StatusOr<std::vector<std::vector<PyArray>>>
PyLoadedExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<const ExecuteShardedArg> args) {
  xla::ifrt::ExecuteOptions options = options_;
  options.fill_status = false;
  std::optional<std::vector<PjRtFuture<>>> returned_futures;
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options, client_, ifrt_loaded_executable_.get(), args,
                          returned_futures));
  return outputs_and_tokens.DisassembleIntoSingleDeviceArrays();
}

absl::StatusOr<std::pair<std::vector<std::vector<PyArray>>, PyShardedToken>>
PyLoadedExecutable::ExecuteShardedOnLocalDevicesWithTokens(
    absl::Span<const ExecuteShardedArg> args) {
  xla::ifrt::ExecuteOptions options = options_;
  options.fill_status = true;
  std::optional<std::vector<PjRtFuture<>>> returned_futures;
  returned_futures.emplace();
  TF_ASSIGN_OR_RETURN(auto outputs_and_tokens,
                      ExecuteShardedOnLocalDevicesInternal(
                          options, client_, ifrt_loaded_executable_.get(), args,
                          returned_futures));
  return std::make_pair(outputs_and_tokens.DisassembleIntoSingleDeviceArrays(),
                        outputs_and_tokens.ConsumeToken());
}

absl::StatusOr<PyExecuteResults> PyLoadedExecutable::ExecuteSharded(
    std::vector<ExecuteShardedArg> args, bool with_tokens) {
  xla::ifrt::ExecuteOptions options = options_;
  options.fill_status = with_tokens;
  std::optional<std::vector<PjRtFuture<>>> returned_futures;
  if (with_tokens) {
    returned_futures.emplace();
  }
  absl::Span<const ExecuteShardedArg> span_args = args;
  return ExecuteShardedOnLocalDevicesInternal(options, client_,
                                              ifrt_loaded_executable_.get(),
                                              span_args, returned_futures);
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
PyLoadedExecutable::HloModules() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetHloModules();
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
PyLoadedExecutable::GetOutputMemoryKinds() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetOutputMemoryKinds();
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtLayout>>>
PyLoadedExecutable::GetParameterLayouts() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetParameterLayouts();
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtLayout>>>
PyLoadedExecutable::GetOutputLayouts() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetOutputLayouts();
}

std::optional<std::vector<OpSharding>>
PyLoadedExecutable::GetParameterShardings() const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetParameterShardings();
}

std::optional<std::vector<OpSharding>> PyLoadedExecutable::GetOutputShardings()
    const {
  nb::gil_scoped_release gil_release;
  return ifrt_loaded_executable_->GetOutputShardings();
}

void PyLoadedExecutable::KeepAlive(nb::object obj) {
  keepalives_.push_back(std::move(obj));
}

}  // namespace xla
