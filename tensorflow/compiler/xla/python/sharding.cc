/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/sharding.h"

#include <utility>

#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/python/util.h"

namespace jax {

namespace py = pybind11;

size_t ShardingHash(const pybind11::object& sharding) {
  auto type = sharding.get_type();

  if (type.is(NamedSharding::type())) {
    const auto* mesh_sharding = xla::fast_cast<jax::NamedSharding>(sharding);
    return absl::Hash<void*>()(mesh_sharding->mesh().ptr());
  }

  if (type.is(OpShardingSharding::type())) {
    auto* op_sharding = xla::fast_cast<OpShardingSharding>(sharding);
    return op_sharding->Hash();
  }

  if (type.is(SingleDeviceSharding::type())) {
    auto* single_device_sharding =
        xla::fast_cast<SingleDeviceSharding>(sharding);
    return absl::Hash<void*>()(single_device_sharding->device().ptr());
  }

  return py::hash(sharding);
}

bool ShardingEqual(const pybind11::object& a, const pybind11::object& b) {
  if (a.ptr() == b.ptr()) return true;

  auto a_type = a.get_type();
  auto b_type = b.get_type();

  if (!a_type.is(b_type)) return false;

  if (a_type.is(NamedSharding::type())) {
    auto* a_mesh_sharding = xla::fast_cast<const NamedSharding>(a);
    auto* b_mesh_sharding = xla::fast_cast<const NamedSharding>(b);

    return a_mesh_sharding->mesh().ptr() == b_mesh_sharding->mesh().ptr() &&
           a_mesh_sharding->spec().equal(b_mesh_sharding->spec());
  }

  if (a_type.is(OpShardingSharding::type())) {
    auto* a_op_sharding_sharding = xla::fast_cast<const OpShardingSharding>(a);
    auto* b_op_sharding_sharding = xla::fast_cast<const OpShardingSharding>(b);

    return a_op_sharding_sharding == b_op_sharding_sharding;
  }

  if (a_type.is(SingleDeviceSharding::type())) {
    auto* a_single_device_sharding =
        xla::fast_cast<const SingleDeviceSharding>(a);
    auto* b_single_device_sharding =
        xla::fast_cast<const SingleDeviceSharding>(b);

    return a_single_device_sharding->device().ptr() ==
           b_single_device_sharding->device().ptr();
  }

  return a.equal(b);
}

NamedSharding::NamedSharding(py::object mesh, py::object spec,
                             py::object parsed_pspec)
    : XLACompatibleSharding(/*num_devices=*/[&mesh]() {
        py::array devices = mesh.attr("devices");
        return devices.size();
      }()),
      mesh_(std::move(mesh)),
      spec_(std::move(spec)),
      parsed_pspec_(std::move(parsed_pspec)) {
  py::cast(this).attr("_preprocess")();
}

void RegisterSharding(py::module& m) {
  py::object abc_module = py::module::import("abc");
  py::object abc_meta = abc_module.attr("ABCMeta");
  py::object abc_init = abc_module.attr("_abc_init");

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<Sharding>(m, "Sharding", py::metaclass(abc_meta));
  abc_init(py::type::of<Sharding>());

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<XLACompatibleSharding, Sharding>(m, "XLACompatibleSharding",
                                              py::metaclass(abc_meta));
  abc_init(py::type::of<XLACompatibleSharding>());

  py::class_<NamedSharding, XLACompatibleSharding>(m, "NamedSharding",
                                                   py::dynamic_attr())
      .def(py::init<py::object, py::object, py::object>(), py::arg("mesh"),
           py::arg("spec"), py::arg("_parsed_pspec") = py::none())
      .def_property_readonly("mesh", &NamedSharding::mesh)
      .def_property_readonly("spec", &NamedSharding::spec)
      .def_property("_parsed_pspec", &NamedSharding::parsed_pspec,
                    &NamedSharding::set_parsed_pspec);

  py::class_<SingleDeviceSharding, XLACompatibleSharding>(
      m, "SingleDeviceSharding", py::dynamic_attr())
      .def(py::init<py::object>(), py::arg("device"))
      .def_property_readonly("_device", &SingleDeviceSharding::device);

  py::class_<PmapSharding, XLACompatibleSharding>(m, "PmapSharding",
                                                  py::dynamic_attr())
      .def(py::init<py::object, ShardingSpec>(), py::arg("devices"),
           py::arg("sharding_spec"))
      .def_property_readonly("devices", &PmapSharding::devices)
      .def_property_readonly("sharding_spec", &PmapSharding::sharding_spec);

  py::class_<OpShardingSharding, XLACompatibleSharding>(m, "OpShardingSharding",
                                                        py::dynamic_attr())
      .def(py::init<py::list, xla::OpSharding>(), py::arg("devices"),
           py::arg("op_sharding"))
      .def(py::init<py::tuple, xla::OpSharding>(), py::arg("devices"),
           py::arg("op_sharding"))
      .def_property_readonly("_devices", &OpShardingSharding::devices)
      .def_property_readonly("_op_sharding", &OpShardingSharding::op_sharding);
}

}  // namespace jax
