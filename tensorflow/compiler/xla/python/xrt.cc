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

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xrt/client/xrt_client.h"
#include "tensorflow/compiler/xrt/client/xrt_grpc_eager_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

namespace py = pybind11;

xla::StatusOr<std::shared_ptr<XrtTfClient>> GetTfClient(const string& address,
                                                        const string& worker) {
  ClusterDef cluster_def;
  JobDef* job = cluster_def.add_job();
  job->set_name(worker);
  (*job->mutable_tasks())[0] = address;
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  TF_ASSIGN_OR_RETURN(std::shared_ptr<GrpcChannelCache> channel_cache,
                      GetGrpcChannelCache(cluster_def, channel_func));
  return std::make_shared<XrtTfClient>(cluster_def, channel_cache);
}

// TODO(phawkins): This function won't produce a particularly good device
// assignment since it knows nothing about the hardware or its topology.
// It's here mostly as a placeholder until we do something smarter.
xla::StatusOr<xla::DeviceAssignment> AssignDevices(int num_replicas,
                                                   int num_computations) {
  return xla::ComputationPlacer().AssignDevices(num_replicas, num_computations);
}

}  // namespace

void AddXrtSubmodule(py::module* module) {
  py::module m = module->def_submodule("xrt", "XRT backend");

  m.def("AssignDevices", &AssignDevices,
        "Computes a default device assignment.");

  py::class_<XrtTfClient, std::shared_ptr<XrtTfClient>> xrt_tf_client(
      m, "XrtTfClient");
  m.def("GetTfClient", &GetTfClient, "Returns a TensorFlow client.");

  py::class_<XrtTfContext::Options>(m, "XrtTfContextOptions")
      .def(py::init<>())
      .def_readwrite("async", &XrtTfContext::Options::async)
      .def_readwrite("max_queue_size", &XrtTfContext::Options::max_queue_size);

  py::class_<XrtTfContext, std::shared_ptr<XrtTfContext>>(m, "XrtTfContext")
      .def_static("Create", &XrtTfContext::Create);

  py::class_<XrtContext, std::shared_ptr<XrtContext>>(m, "XrtContext")
      .def_static("Create", &XrtContext::Create)
      .def("DeviceCount", &XrtContext::device_count)
      .def_property_readonly("tf_device_ids", &XrtContext::tf_device_ids);

  py::class_<XrtBuffer, std::shared_ptr<XrtBuffer>>(m, "XrtBuffer")
      .def_static("FromLiteral", &XrtBuffer::FromLiteral)
      .def("ToPython",
           [](std::shared_ptr<XrtBuffer> buffer) -> xla::StatusOr<py::object> {
             auto literal = absl::make_unique<xla::Literal>();
             {
               py::gil_scoped_release gil_release;
               TF_ASSIGN_OR_RETURN(*literal, buffer->ToLiteral());
             }
             return xla::LiteralToPython(std::move(literal));
           })
      .def("Delete", &XrtBuffer::Delete)
      .def("DestructureTuple", &XrtBuffer::DestructureTuple);

  py::class_<XrtExecutable, std::shared_ptr<XrtExecutable>>(m, "XrtExecutable")
      .def_static("Compile",
                  [](std::shared_ptr<XrtContext> context,
                     const std::string& hlo_module_proto_serialized,
                     const std::vector<xla::Shape>& argument_shapes,
                     const xla::Shape& result_shape,
                     const xla::DeviceAssignment& device_assignment) {
                    xla::HloModuleProto hlo_module_proto;
                    hlo_module_proto.ParsePartialFromString(
                        hlo_module_proto_serialized);
                    return XrtExecutable::Compile(context, hlo_module_proto,
                                                  argument_shapes, result_shape,
                                                  device_assignment);
                  })
      .def("Execute", &XrtExecutable::Execute)
      .def("ExecuteReplicated",
           [](XrtExecutable& executable,
              std::vector<std::vector<std::vector<std::shared_ptr<XrtBuffer>>>>
                  pyargs)
               -> xla::StatusOr<
                   std::vector<std::vector<std::shared_ptr<XrtBuffer>>>> {
             const xla::DeviceAssignment& device_assignment =
                 executable.device_assignment();
             if (pyargs.size() != device_assignment.computation_count()) {
               return xla::InvalidArgument(
                   "Outermost argument list must have one entry per "
                   "computation; "
                   "got %d args, device assignment has %d computations.",
                   pyargs.size(), device_assignment.computation_count());
             }
             std::vector<xla::Array2D<std::shared_ptr<XrtBuffer>>> args(
                 pyargs.size());
             for (int i = 0; i < pyargs.size(); ++i) {
               if (pyargs[i].size() != device_assignment.replica_count() ||
                   pyargs[i].empty()) {
                 return xla::InvalidArgument(
                     "Mismatch in number of replicas; got %d arguments, but "
                     "device assignment has %d replicas.",
                     pyargs[i].size(), device_assignment.replica_count());
               }

               int arg_count = pyargs[i][0].size();
               args[i] = xla::Array2D<std::shared_ptr<XrtBuffer>>(
                   device_assignment.replica_count(), arg_count);
               for (int j = 0; j < pyargs[i].size(); ++j) {
                 if (pyargs[i][j].size() != arg_count) {
                   return xla::InvalidArgument(
                       "Mismatched number of arguments to computation %d for "
                       "different replicas; %d vs %d arguments.",
                       i, arg_count, pyargs[i][j].size());
                 }
                 for (int k = 0; k < arg_count; ++k) {
                   args[i](j, k) = pyargs[i][j][k];
                 }
               }
             }

             TF_ASSIGN_OR_RETURN(auto result,
                                 executable.ExecuteReplicated(args));
             std::vector<std::vector<std::shared_ptr<XrtBuffer>>> pyresult(
                 result.n1());
             for (int i = 0; i < result.n1(); ++i) {
               pyresult[i].resize(result.n2());
               for (int j = 0; j < result.n2(); ++j) {
                 pyresult[i][j] = result(i, j);
               }
             }
             return pyresult;
           })
      .def("Delete", &XrtExecutable::Delete)
      .def("DeviceOrdinals", [](const XrtExecutable& executable) {
        return std::vector<int>(executable.device_assignment().begin(),
                                executable.device_assignment().end());
      });

  m.doc() = "XRT backend plugin";
}

}  // namespace tensorflow
