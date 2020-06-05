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

#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/traceback_manager.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {

namespace py = pybind11;

PyClient::PyClient(std::shared_ptr<PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)) {}

std::vector<ClientAndPtr<Device>> PyClient::Devices() {
  std::vector<ClientAndPtr<Device>> devices;
  devices.reserve(pjrt_client_->devices().size());
  for (const auto& device : pjrt_client_->devices()) {
    devices.push_back(WrapWithClient(shared_from_this(), device.get()));
  }
  return devices;
}

std::vector<ClientAndPtr<Device>> PyClient::LocalDevices() {
  std::vector<ClientAndPtr<Device>> devices;
  devices.reserve(pjrt_client_->local_devices().size());
  for (Device* device : pjrt_client_->local_devices()) {
    devices.push_back(WrapWithClient(shared_from_this(), device));
  }
  return devices;
}

StatusOr<std::vector<std::vector<ClientAndPtr<Device>>>>
PyClient::GetDefaultDeviceAssignment(int num_replicas, int num_partitions) {
  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      pjrt_client_->GetDefaultDeviceAssignment(num_replicas, num_partitions));
  std::vector<std::vector<ClientAndPtr<Device>>> result;
  result.resize(num_replicas);
  for (int r = 0; r < num_replicas; ++r) {
    result[r].resize(num_partitions);
    for (int p = 0; p < num_partitions; ++p) {
      int device_id = device_assignment(r, p);
      auto iter = pjrt_client_->id_to_device().find(device_id);
      CHECK(iter != pjrt_client_->id_to_device().end()) << device_id;
      result[r][p] = WrapWithClient(shared_from_this(), iter->second);
    }
  }
  return result;
}

StatusOr<std::vector<ClientAndPtr<Device>>>
PyClient::GetDefaultDeviceAssignment1D(int num_replicas) {
  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      pjrt_client_->GetDefaultDeviceAssignment(
                          num_replicas, /*num_partitions=*/1));
  std::vector<ClientAndPtr<Device>> result;
  for (int i = 0; i < num_replicas; ++i) {
    int device_id = device_assignment(i, 0);
    auto iter = pjrt_client_->id_to_device().find(device_id);
    CHECK(iter != pjrt_client_->id_to_device().end()) << device_id;
    result.push_back(WrapWithClient(shared_from_this(), iter->second));
  }
  return result;
}

StatusOr<std::unique_ptr<PyBuffer>> PyClient::BufferFromPyal(
    const pybind11::object& argument, Device* device, bool force_copy) {
  if (device == nullptr) {
    TF_RET_CHECK(!pjrt_client_->local_devices().empty());
    device = pjrt_client_->local_devices().front();
  }
  CHECK(device != nullptr);
  auto iter = pjrt_client_->id_to_device().find(device->id());
  if (iter->second != device) {
    return InvalidArgument("Cannot copy value to device '%s' with '%s' backend",
                           device->DebugString(),
                           pjrt_client_->platform_name());
  }
  GlobalPyRefManager()->CollectGarbage();

  absl::optional<CastToArrayResult> c = CastToArray(argument);
  if (!c) {
    return InvalidArgument("from_python argument must be an array.");
  }

  TF_ASSIGN_OR_RETURN(PythonBufferTree tree, GetPythonBufferTree(argument));
  std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref =
      GlobalPyRefManager()->ManageReference(std::move(c->array));

  auto traceback = TracebackManager::Get()->GetTraceback();

  py::gil_scoped_release gil_release;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer> buffer,
      PjRtBuffer::FromHostBuffer(c->buf_ptr, c->shape, force_copy,
                                 std::move(py_buffer_ref), pjrt_client_.get(),
                                 device));
  return std::make_unique<PyBuffer>(shared_from_this(), std::move(buffer),
                                    traceback);
}

StatusOr<std::unique_ptr<PyExecutable>> PyClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  auto traceback = TracebackManager::Get()->GetTraceback();
  py::gil_scoped_release gil_release;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      PjRtExecutable::Compile(computation, pjrt_client_.get(),
                                              std::move(options)));
  return std::make_unique<PyExecutable>(
      shared_from_this(), std::move(executable), std::move(traceback));
}

}  // namespace xla
