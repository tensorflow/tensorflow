/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/function/runtime_client/runtime_client.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

PYBIND11_MAKE_OPAQUE(tensorflow::EagerContext);

PYBIND11_MODULE(runtime_client_pybind, m) {
  pybind11::class_<tensorflow::EagerContext, tensorflow::EagerContextPtr>
      EagerContext(m, "EagerContext");

  m.def("GlobalEagerContext", &tensorflow::core::function::GlobalEagerContext,
        pybind11::return_value_policy::reference);

  m.def("GlobalPythonEagerContext",
        &tensorflow::core::function::GlobalPythonEagerContext,
        pybind11::return_value_policy::reference);

  pybind11::class_<tensorflow::core::function::Runtime>(m, "Runtime")
      .def(pybind11::init<tensorflow::EagerContext&>())
      // TODO(mdan): Rename to GetFunctionProto once pybind11_protobuf available
      .def(
          "GetFunctionProtoString",
          [](tensorflow::core::function::Runtime& r, const std::string& name) {
            return pybind11::bytes(
                r.GetFunctionProto(name)->SerializeAsString());
          },
          pybind11::return_value_policy::reference)
      // TODO(mdan): Rename to CreateFunction once pybind11_protobuf available
      .def("CreateFunctionFromString",
           [](tensorflow::core::function::Runtime& r, const std::string& def) {
             tensorflow::FunctionDef proto;
             proto.ParseFromString(def);
             return r.CreateFunction(proto);
           })
      .def("TransformFunction",
           &tensorflow::core::function::Runtime::TransformFunction);
}
