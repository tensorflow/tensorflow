/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {

namespace py = pybind11;

PYBIND11_MODULE(tpu_client_extension, m) {
  // Initializes the NumPy API for the use of the types module.
  if (!InitializeNumpyAPIForTypes()) {
    throw std::runtime_error("Unable to initialize Numpy API");
  }

  py::class_<PyTpuClient, std::shared_ptr<PyTpuClient>>(m, "TpuClient")
      .def_static("Get", &PyTpuClient::Get, py::arg("worker"))
      .def_property_readonly("platform", &PyTpuClient::platform_name)
      .def("device_count", &PyTpuClient::device_count)
      .def("local_device_count", &PyTpuClient::local_device_count)
      .def("devices", &PyTpuClient::devices)
      .def("local_devices", &PyTpuClient::local_devices)
      .def("host_id", &PyTpuClient::host_id)
      .def("get_default_device_assignment",
           [](PyTpuClient* client, int num_replicas, int num_partitions)
               -> StatusOr<std::vector<std::vector<std::shared_ptr<Device>>>> {
             TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                                 client->GetDefaultDeviceAssignment(
                                     num_replicas, num_partitions));
             std::vector<std::vector<std::shared_ptr<Device>>> result;
             result.resize(num_replicas);
             for (int r = 0; r < num_replicas; ++r) {
               result[r].resize(num_partitions);
               for (int p = 0; p < num_partitions; ++p) {
                 int device_id = device_assignment(r, p);
                 auto iter = client->id_to_device().find(device_id);
                 CHECK(iter != client->id_to_device().end()) << device_id;
                 result[r][p] = iter->second;
               }
             }
             return result;
           })
      // TODO(skye): delete after all callers can handle 2D output
      .def("get_default_device_assignment",
           [](PyTpuClient* client, int num_replicas)
               -> StatusOr<std::vector<std::shared_ptr<Device>>> {
             TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                                 client->GetDefaultDeviceAssignment(
                                     num_replicas, /*num_partitions=*/1));
             std::vector<std::shared_ptr<Device>> result;
             for (int i = 0; i < num_replicas; ++i) {
               int device_id = device_assignment(i, 0);
               auto iter = client->id_to_device().find(device_id);
               CHECK(iter != client->id_to_device().end()) << device_id;
               result.push_back(iter->second);
             }
             return result;
           })
      .def("transfer_to_infeed",
           [](PyTpuClient* client, const LiteralSlice& literal,
              int device_ordinal) {
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             return client->TransferToInfeed(literal, device_ordinal);
           })
      .def("transfer_from_outfeed",
           [](PyTpuClient* client, const Shape& shape,
              int device_ordinal) -> StatusOr<py::object> {
             GlobalPyRefManager()->CollectGarbage();
             std::shared_ptr<Literal> literal_shared;
             {
               py::gil_scoped_release gil_release;
               TF_ASSIGN_OR_RETURN(Literal literal, client->TransferFromOutfeed(
                                                        shape, device_ordinal));
               literal_shared = std::make_shared<Literal>(std::move(literal));
             }
             return LiteralToPython(std::move(literal_shared));
           })
      .def(
          "buffer_from_pyval",
          [](std::shared_ptr<PyTpuClient> client,
             const pybind11::object& argument, std::shared_ptr<Device> device,
             bool force_copy) -> StatusOr<std::unique_ptr<PyTpuBuffer>> {
            if (device == nullptr) {
              TF_RET_CHECK(!client->local_devices().empty());
              device = client->local_devices().front();
            }
            auto iter = client->id_to_device().find(device->id());
            if (iter->second != device) {
              return InvalidArgument(
                  "Cannot copy value to device '%s' with '%s' backend",
                  device->DebugString(), client->platform_name());
            }
            GlobalPyRefManager()->CollectGarbage();
            TF_ASSIGN_OR_RETURN(PythonBufferTree tree,
                                GetPythonBufferTree(argument));
            std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref =
                GlobalPyRefManager()->ManageReferences(
                    absl::MakeSpan(tree.arrays));
            tree.arrays.clear();

            std::vector<BorrowingLiteral> leaves;
            leaves.insert(leaves.end(),
                          std::make_move_iterator(tree.leaves.begin()),
                          std::make_move_iterator(tree.leaves.end()));

            py::gil_scoped_release gil_release;
            return PyTpuBuffer::FromLiterals(
                std::move(leaves), tree.shape, std::move(py_buffer_ref),
                std::move(client), std::move(device));
          },
          py::arg("argument"), py::arg("device") = nullptr,
          py::arg("force_copy") = false)
      .def(
          "compile",
          [](std::shared_ptr<PyTpuClient> client,
             const XlaComputation& computation, CompileOptions options)
              -> StatusOr<std::unique_ptr<PyTpuExecutable>> {
            py::gil_scoped_release gil_release;
            return PyTpuExecutable::Compile(
                computation, options.argument_layouts,
                &options.executable_build_options, client,
                options.parameter_is_tupled_arguments);
          },
          py::arg("computation"),
          py::arg("compile_options") = CompileOptions());

  py::class_<PyTpuBuffer>(m, "PyTpuBuffer")
      .def_property_readonly("client", &PyTpuBuffer::client)
      .def("copy_to_device",
           [](PyTpuBuffer* buffer, std::shared_ptr<Device> dst_device) {
             CHECK(dst_device != nullptr);
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             return buffer->CopyToDevice(std::move(dst_device));
           })
      .def("delete", &PyTpuBuffer::Delete)
      .def("block_host_until_ready",
           [](PyTpuBuffer* buffer) {
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             return buffer->BlockHostUntilReady();
           })
      .def("copy_to_host_async", &PyTpuBuffer::CopyToHostAsync,
           py::call_guard<py::gil_scoped_release>())
      .def("to_py",
           [](PyTpuBuffer* buffer) -> StatusOr<py::object> {
             GlobalPyRefManager()->CollectGarbage();
             std::shared_ptr<Literal> literal;
             {
               py::gil_scoped_release gil_release;
               TF_ASSIGN_OR_RETURN(literal, buffer->ToLiteral());
             }
             return LiteralToPython(std::move(literal));
           })
      .def("shape", &PyTpuBuffer::on_host_shape)
      .def("device", &PyTpuBuffer::device)
      .def("platform", &PyTpuBuffer::platform_name)
      .def("is_deleted", [](const PyTpuBuffer& buffer) {
        return buffer.DeviceBuffer() == nullptr;
      });

  py::class_<PyTpuExecutable>(m, "TpuExecutable")
      .def("local_logical_device_ids",
           &PyTpuExecutable::local_logical_device_ids)
      .def("local_devices", &PyTpuExecutable::local_devices)
      .def_property_readonly("client", &PyTpuExecutable::client)
      .def("size_of_generated_code_in_bytes",
           &PyTpuExecutable::SizeOfGeneratedCodeInBytes)
      .def("Delete", &PyTpuExecutable::Delete)
      .def("Execute", &PyTpuExecutable::Execute,
           py::call_guard<py::gil_scoped_release>(), py::arg("arguments"))
      .def("ExecuteOnLocalDevices", &PyTpuExecutable::ExecuteOnLocalDevices,
           py::call_guard<py::gil_scoped_release>(), py::arg("arguments"))
      .def("delete", &PyTpuExecutable::Delete)
      .def("execute", &PyTpuExecutable::Execute,
           py::call_guard<py::gil_scoped_release>(), py::arg("arguments"))
      .def("execute_on_local_devices", &PyTpuExecutable::ExecuteOnLocalDevices,
           py::call_guard<py::gil_scoped_release>(), py::arg("arguments"));

  py::class_<TpuDevice, Device, std::shared_ptr<TpuDevice>>(m, "TpuDevice")
      .def_property_readonly("coords", &TpuDevice::coords)
      .def_property_readonly("core_on_chip", &TpuDevice::core_on_chip)
      .def("__repr__", [](const TpuDevice& device) {
        return absl::StrFormat(
            "TpuDevice(id=%i, host_id=%i, coords=(%i,%i,%i), core_on_chip=%i)",
            device.id(), device.host_id(), device.coords()[0],
            device.coords()[1], device.coords()[2], device.core_on_chip());
      });
}  // NOLINT(readability/fn_size)

}  // namespace xla
