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

#include "tensorflow/compiler/xla/python/pmap_lib.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/python/absl_casters.h"
#include "tensorflow/core/platform/logging.h"

namespace jax {

namespace py = pybind11;

// TODO(jblespiau): Using `NoSharding` instead of `None` would allow us to
// simplify the conversion logic.
std::vector<AvalDimSharding> PyShardingToCpp(pybind11::tuple py_sharding) {
  std::vector<AvalDimSharding> cpp_sharding;
  cpp_sharding.reserve(py_sharding.size());
  for (py::handle value : py_sharding) {
    if (value.is_none()) {
      cpp_sharding.push_back(NoSharding());
    } else if (py::isinstance<Chunked>(value)) {
      cpp_sharding.push_back(py::cast<Chunked>(value));
    } else if (py::isinstance<Unstacked>(value)) {
      cpp_sharding.push_back(py::cast<Unstacked>(value));
    } else {
      throw std::runtime_error(
          absl::StrCat("Not supported Python object in PyShardingToCpp in "
                       "pmap_lib.cc. The object was of type ",
                       py::cast<std::string>(py::str(value.get_type())),
                       "\n:", py::cast<std::string>(py::str(value))));
    }
  }
  return cpp_sharding;
}

pybind11::tuple CppShardingToPy(std::vector<AvalDimSharding> sharding) {
  py::tuple result(sharding.size());
  int counter = 0;
  for (auto value : sharding) {
    if (absl::holds_alternative<NoSharding>(value)) {
      result[counter++] = py::none();
    } else if (absl::holds_alternative<Chunked>(value)) {
      py::handle handle = py::cast(absl::get<Chunked>(value));
      result[counter++] = py::cast<py::object>(handle);
    } else if (absl::holds_alternative<Unstacked>(value)) {
      py::handle handle = py::cast(absl::get<Unstacked>(value));
      result[counter++] = py::cast<py::object>(handle);
    } else {
      LOG(FATAL) << "Unhandled CPP type in CppShardingToPy.";
    }
  }
  return result;
}

std::vector<MeshDimAssignment> PyMeshShardingToCpp(
    pybind11::tuple py_mesh_mapping) {
  return py::cast<std::vector<MeshDimAssignment>>(py_mesh_mapping);
}

pybind11::tuple CppMeshMappingToPy(
    std::vector<MeshDimAssignment> mesh_mapping) {
  py::tuple result(mesh_mapping.size());
  int counter = 0;
  for (auto& value : mesh_mapping) {
    result[counter] = py::cast(value);
    ++counter;
  }
  return result;
}

void BuildPmapSubmodule(pybind11::module& m) {
  py::module pmap_lib = m.def_submodule("pmap_lib", "Jax C++ pmap library");

  py::class_<NoSharding> no_sharding(pmap_lib, "NoSharding");
  no_sharding.def(py::init<>())
      .def("__repr__",
           [](const NoSharding& chuncked) { return "NoSharding()"; })
      .def("__eq__", [](const NoSharding& self, py::object obj) {
        return py::isinstance<NoSharding>(obj);
      });

  py::class_<Chunked> chunked(pmap_lib, "Chunked");
  chunked.def(py::init<std::vector<int>>())
      .def_readonly("chunks", &Chunked::chunks)
      .def_readonly("num_chunks", &Chunked::chunks)
      .def("__repr__",
           [](const Chunked& chuncked) {
             return absl::StrCat("Chunked(",
                                 absl::StrJoin(chuncked.chunks, ","), ")");
           })
      .def("__eq__", [](const Chunked& self, py::object other) {
        if (!py::isinstance<Chunked>(other)) {
          return false;
        }
        return self == py::cast<const Chunked&>(other);
      });

  py::class_<Unstacked> unstacked(pmap_lib, "Unstacked");
  unstacked.def(py::init<int>())
      .def_readonly("size", &Unstacked::size)
      .def("__repr__",
           [](const Unstacked& x) {
             return absl::StrCat("Unstacked(", x.size, ")");
           })
      .def("__eq__", [](const Unstacked& self, py::object other) {
        if (!py::isinstance<Unstacked>(other)) {
          return false;
        }
        return self == py::cast<const Unstacked&>(other);
      });

  py::class_<ShardedAxis> sharded_axis(pmap_lib, "ShardedAxis");
  sharded_axis.def(py::init<int>()).def_readonly("axis", &ShardedAxis::axis);
  sharded_axis
      .def("__repr__",
           [](const ShardedAxis& x) {
             return absl::StrCat("ShardedAxis(axis=", x.axis, ")");
           })
      .def("__eq__", [](const ShardedAxis& self, const ShardedAxis& other) {
        return self == other;
      });

  py::class_<Replicated> replicated(pmap_lib, "Replicated");
  replicated.def(py::init<int>())
      .def_readonly("replicas", &Replicated::replicas)
      .def("__repr__",
           [](const Replicated& x) {
             return absl::StrCat("Replicated(replicas=", x.replicas, ")");
           })
      .def("__eq__", [](const Replicated& self, const Replicated& other) {
        return self == other;
      });

  py::class_<ShardingSpec> sharding_spec(pmap_lib, "ShardingSpec");
  sharding_spec
      .def(py::init<py::tuple, py::tuple>(), py::arg("sharding"),
           py::arg("mesh_mapping"))
      .def_property_readonly("sharding", &ShardingSpec::GetPySharding)
      .def_property_readonly("mesh_mapping", &ShardingSpec::GetPyMeshMapping);

  py::class_<ShardedDeviceArray> sda(pmap_lib, "ShardedDeviceArray");
  sda.def(py::init<pybind11::handle, ShardingSpec, pybind11::list>())
      .def_property_readonly("aval", &ShardedDeviceArray::GetAval)
      .def_property_readonly("sharding_spec",
                             &ShardedDeviceArray::GetShardingSpec)
      .def_property_readonly("device_buffers",
                             &ShardedDeviceArray::GetDeviceBuffers);
}

}  // namespace jax
