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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/py_values.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/core/profiler/profile.pb.h"

namespace xla {

namespace py = pybind11;
namespace pprof = tensorflow::tfprof::pprof;

PyClient::PyClient(std::unique_ptr<PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)) {}
PyClient::PyClient(std::shared_ptr<PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)) {}

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

std::vector<ClientAndPtr<PyBuffer>> PyClient::LiveBuffers() {
  CHECK(PyGILState_Check());
  std::vector<ClientAndPtr<PyBuffer>> buffers;
  for (PyBuffer* buffer = buffers_; buffer; buffer = buffer->next_) {
    if (!buffer->is_deleted()) {
      buffers.push_back(WrapWithClient(shared_from_this(), buffer));
    }
  }
  return buffers;
}

Status PyClient::Defragment() {
  CHECK(PyGILState_Check());
  absl::flat_hash_set<PjRtBuffer*> buffer_set;
  for (PyBuffer* buffer = buffers_; buffer; buffer = buffer->next_) {
    if (!buffer->is_deleted()) {
      buffer_set.insert(buffer->buffer());
    }
  }
  std::vector<PjRtBuffer*> buffers(buffer_set.begin(), buffer_set.end());

  std::vector<PjRtExecutable*> execs;
  for (PyExecutable* exec = executables_; exec; exec = exec->next_) {
    execs.push_back(exec->mutable_pjrt_executable());
  }
  return pjrt_client_->Defragment(buffers, execs);
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
  options.force_lazy_arrays = true;
  TF_ASSIGN_OR_RETURN(DevicePutResult put,
                      DevicePut(argument, device, options));

  if (put.owned_buffer) {
    auto traceback = Traceback::Get();
    return py::cast(std::make_unique<PyBuffer>(
        shared_from_this(), std::move(put.owned_buffer), std::move(traceback)));
  } else {
    return py::reinterpret_borrow<py::object>(put.owning_pybuffer);
  }
}

StatusOr<std::shared_ptr<PyExecutable>> PyClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  std::unique_ptr<PjRtExecutable> executable;
  absl::optional<std::string> fingerprint;
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
      std::move(fingerprint));
}

class ProfileBuilder {
 public:
  ProfileBuilder();
  pprof::Profile& profile() { return profile_; }

  // Adds or returns the ID of `s` in the table.
  int StringId(const std::string& s);

  // Adds or returns the ID of a function.
  int FunctionId(PyCodeObject* code);

  // Adds or returns the ID of a code location.
  int LocationId(PyCodeObject* code, int instruction);

 private:
  pprof::Profile profile_;

  absl::flat_hash_map<std::string, int> strings_;
  absl::flat_hash_map<PyCodeObject*, int> functions_;
  absl::flat_hash_map<std::pair<PyCodeObject*, int>, int> locations_;
};

ProfileBuilder::ProfileBuilder() { CHECK_EQ(0, StringId("")); }

int ProfileBuilder::StringId(const std::string& s) {
  auto ret = strings_.emplace(s, profile_.string_table_size());
  if (ret.second) {
    profile_.add_string_table(s);
  }
  return ret.first->second;
}

int ProfileBuilder::FunctionId(PyCodeObject* code) {
  // +1 because id 0 is reserved.
  auto ret = functions_.emplace(code, profile_.function_size() + 1);
  if (ret.second) {
    auto* function = profile_.add_function();
    function->set_id(ret.first->second);
    int name = StringId(py::str(code->co_name));
    function->set_name(name);
    function->set_system_name(name);
    function->set_filename(StringId(py::str(code->co_filename)));
    function->set_start_line(code->co_firstlineno);
  }
  return ret.first->second;
}

int ProfileBuilder::LocationId(PyCodeObject* code, int instruction) {
  // +1 because id 0 is reserved.
  auto ret = locations_.emplace(std::make_pair(code, instruction),
                                profile_.location_size() + 1);
  if (ret.second) {
    auto* location = profile_.add_location();
    location->set_id(ret.first->second);
    auto* line = location->add_line();
    line->set_function_id(FunctionId(code));
    line->set_line(PyCode_Addr2Line(code, instruction));
  }
  return ret.first->second;
}

namespace {

struct HeapProfileKey {
  Traceback* traceback;
  int64 size;
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
    h = H::combine_contiguous(std::move(h), key.traceback->raw_frames().begin(),
                              key.traceback->raw_frames().size());
  }
  h = H::combine(std::move(h), key.size, key.device);
  return h;
}

}  // namespace

py::bytes PyClient::HeapProfile() {
  CHECK(PyGILState_Check());
  absl::flat_hash_set<PjRtBuffer*> buffer_set;
  absl::flat_hash_map<HeapProfileKey, int64> entries;
  for (PyBuffer* buffer = buffers_; buffer; buffer = buffer->next_) {
    // We only wish to count each PjRtBuffer once, even though they may be
    // shared by multiple PyBuffers.
    if (buffer_set.insert(buffer->buffer()).second) {
      HeapProfileKey key{buffer->traceback(),
                         buffer->buffer()->OnDeviceSizeInBytes(),
                         buffer->buffer()->device()};
      ++entries[key];
    }
  }

  for (PyExecutable* executable = executables_; executable;
       executable = executable->next_) {
    HeapProfileKey key{executable->traceback(),
                       executable->SizeOfGeneratedCodeInBytes(), nullptr};
    ++entries[key];
  }

  ProfileBuilder builder;
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
  return builder.profile().SerializeAsString();
}

}  // namespace xla
