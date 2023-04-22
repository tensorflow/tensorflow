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

#include "Python.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/detail/common.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_server_lib, m) {
  py::class_<tensorflow::data::DispatchGrpcDataServer>(m,
                                                       "DispatchGrpcDataServer")
      .def("start", &tensorflow::data::DispatchGrpcDataServer::Start)
      .def("stop", &tensorflow::data::DispatchGrpcDataServer::Stop)
      .def("join", &tensorflow::data::DispatchGrpcDataServer::Join,
           py::call_guard<py::gil_scoped_release>())
      .def("bound_port", &tensorflow::data::DispatchGrpcDataServer::BoundPort)
      .def("num_workers",
           [](tensorflow::data::DispatchGrpcDataServer* server) -> int {
             int num_workers;
             tensorflow::Status status = server->NumWorkers(&num_workers);
             tensorflow::MaybeRaiseFromStatus(status);
             return num_workers;
           });

  py::class_<tensorflow::data::WorkerGrpcDataServer>(m, "WorkerGrpcDataServer")
      .def("start", &tensorflow::data::WorkerGrpcDataServer::Start)
      .def("stop", &tensorflow::data::WorkerGrpcDataServer::Stop)
      .def("join", &tensorflow::data::WorkerGrpcDataServer::Join,
           py::call_guard<py::gil_scoped_release>())
      .def("bound_port", &tensorflow::data::WorkerGrpcDataServer::BoundPort)
      .def("num_tasks",
           [](tensorflow::data::WorkerGrpcDataServer* server) -> int {
             int num_tasks;
             tensorflow::Status status = server->NumTasks(&num_tasks);
             tensorflow::MaybeRaiseFromStatus(status);
             return num_tasks;
           });

  m.def(
      "TF_DATA_NewDispatchServer",
      [](std::string serialized_dispatcher_config)
          -> std::unique_ptr<tensorflow::data::DispatchGrpcDataServer> {
        tensorflow::data::experimental::DispatcherConfig config;
        if (!config.ParseFromString(serialized_dispatcher_config)) {
          tensorflow::MaybeRaiseFromStatus(tensorflow::errors::InvalidArgument(
              "Failed to deserialize dispatcher config."));
        }
        std::unique_ptr<tensorflow::data::DispatchGrpcDataServer> server;
        tensorflow::Status status =
            tensorflow::data::NewDispatchServer(config, server);
        tensorflow::MaybeRaiseFromStatus(status);
        return server;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_DATA_NewWorkerServer",
      [](std::string serialized_worker_config)
          -> std::unique_ptr<tensorflow::data::WorkerGrpcDataServer> {
        tensorflow::data::experimental::WorkerConfig config;
        if (!config.ParseFromString(serialized_worker_config)) {
          tensorflow::MaybeRaiseFromStatus(tensorflow::errors::InvalidArgument(
              "Failed to deserialize worker config."));
        }
        std::unique_ptr<tensorflow::data::WorkerGrpcDataServer> server;
        tensorflow::Status status =
            tensorflow::data::NewWorkerServer(config, server);
        tensorflow::MaybeRaiseFromStatus(status);
        return server;
      },
      py::return_value_policy::reference);

  m.def(
      "TF_DATA_GetElementSpec",
      [](tensorflow::int64 dataset_id, const std::string& address,
         const std::string& protocol) -> py::bytes {
        std::string element_spec;
        tensorflow::data::DataServiceDispatcherClient client(address, protocol);
        tensorflow::int64 deadline_micros = tensorflow::kint64max;
        tensorflow::Status status;
        Py_BEGIN_ALLOW_THREADS;
        status = tensorflow::data::grpc_util::Retry(
            [&]() { return client.GetElementSpec(dataset_id, element_spec); },
            /*description=*/
            tensorflow::strings::StrCat(
                "get the element_spec with dispatcher at ", address),
            deadline_micros);
        Py_END_ALLOW_THREADS;
        tensorflow::MaybeRaiseFromStatus(status);
        return py::bytes(element_spec);
      },
      py::return_value_policy::reference);
};
