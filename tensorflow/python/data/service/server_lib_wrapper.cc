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
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_server_lib, m) {
  py::class_<tensorflow::data::MasterGrpcDataServer>(m, "MasterGrpcDataServer");
  py::class_<tensorflow::data::WorkerGrpcDataServer>(m, "WorkerGrpcDataServer");

  m.def(
      "TF_DATA_NewMasterServer",
      [](int port, std::string protocol)
          -> std::unique_ptr<tensorflow::data::MasterGrpcDataServer> {
        std::unique_ptr<tensorflow::data::MasterGrpcDataServer> server;
        tensorflow::Status status =
            tensorflow::data::NewMasterServer(port, protocol, &server);
        tensorflow::MaybeRaiseFromStatus(status);
        server->Start();
        return server;
      },
      py::return_value_policy::reference);
  m.def(
      "TF_DATA_MasterServerBoundPort",
      [](tensorflow::data::MasterGrpcDataServer* server) -> int {
        return server->BoundPort();
      },
      py::return_value_policy::copy);
  m.def("TF_DATA_DeleteMasterServer",
        [](tensorflow::data::MasterGrpcDataServer* server) { server->Stop(); });
  m.def(
      "TF_DATA_MasterServerNumTasks",
      [](tensorflow::data::MasterGrpcDataServer* server) -> int {
        int num_tasks;
        tensorflow::Status status = server->NumTasks(&num_tasks);
        tensorflow::MaybeRaiseFromStatus(status);
        return num_tasks;
      },
      py::return_value_policy::copy);

  m.def(
      "TF_DATA_NewWorkerServer",
      [](int port, std::string protocol, std::string master_address,
         std::string worker_address)
          -> std::unique_ptr<tensorflow::data::WorkerGrpcDataServer> {
        std::unique_ptr<tensorflow::data::WorkerGrpcDataServer> server;
        tensorflow::Status status = tensorflow::data::NewWorkerServer(
            port, protocol, master_address, worker_address, &server);
        tensorflow::MaybeRaiseFromStatus(status);
        server->Start();
        return server;
      },
      py::return_value_policy::reference);
  m.def(
      "TF_DATA_WorkerServerBoundPort",
      [](tensorflow::data::WorkerGrpcDataServer* server) -> int {
        return server->BoundPort();
      },
      py::return_value_policy::copy);
  m.def("TF_DATA_DeleteWorkerServer",
        [](tensorflow::data::WorkerGrpcDataServer* server) { server->Stop(); });
};
