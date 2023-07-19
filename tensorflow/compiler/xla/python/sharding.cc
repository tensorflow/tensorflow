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

#include <string>
#include <utility>

#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/python/util.h"

namespace jax {

namespace py = pybind11;

int Sharding::SafeNumDevices(pybind11::handle sharding) {
  // Pure python shardings are not initialized, so we should not
  // even be casting if they are not initialized.
  bool is_safe_to_cast = [&]() {
    if (!xla::is_pybind_reinterpret_cast_ok<jax::Sharding>(sharding)) {
      return false;
    }
    auto* instance =
        reinterpret_cast<pybind11::detail::instance*>(sharding.ptr());
    for (auto vh : pybind11::detail::values_and_holders(instance)) {
      if (!vh.holder_constructed()) {
        return false;
      }
    }

    return true;
  }();

  if (is_safe_to_cast) {
    auto* cpp_sharding = sharding.cast<jax::Sharding*>();
    if (cpp_sharding->num_devices_.has_value()) {
      return (*cpp_sharding->num_devices_);
    }
  }

  pybind11::set device_set = sharding.attr("device_set");
  return device_set.size();
}

size_t ShardingHash(const pybind11::object& sharding) {
  auto type = sharding.get_type();

  if (type.is(NamedSharding::type())) {
    const auto* named_sharding = xla::fast_cast<jax::NamedSharding>(sharding);
    return absl::Hash<void*>()(named_sharding->mesh().ptr());
  }

  if (type.is(GSPMDSharding::type())) {
    auto* gspmd_sharding = xla::fast_cast<GSPMDSharding>(sharding);
    return gspmd_sharding->Hash();
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
    auto* a_named_sharding = xla::fast_cast<const NamedSharding>(a);
    auto* b_named_sharding = xla::fast_cast<const NamedSharding>(b);

    return a_named_sharding->mesh().ptr() == b_named_sharding->mesh().ptr() &&
           a_named_sharding->spec().equal(b_named_sharding->spec()) &&
           a_named_sharding->memory_kind().equal(
               b_named_sharding->memory_kind());
  }

  if (a_type.is(GSPMDSharding::type())) {
    auto* a_gspmd_sharding = xla::fast_cast<const GSPMDSharding>(a);
    auto* b_gspmd_sharding = xla::fast_cast<const GSPMDSharding>(b);

    return a_gspmd_sharding == b_gspmd_sharding;
  }

  if (a_type.is(SingleDeviceSharding::type())) {
    auto* a_single_device_sharding =
        xla::fast_cast<const SingleDeviceSharding>(a);
    auto* b_single_device_sharding =
        xla::fast_cast<const SingleDeviceSharding>(b);

    return a_single_device_sharding->device().ptr() ==
               b_single_device_sharding->device().ptr() &&
           a_single_device_sharding->memory_kind().equal(
               b_single_device_sharding->memory_kind());
  }

  return a.equal(b);
}

xla::ClientAndPtr<xla::PjRtMemorySpace> GetMemory(
    const xla::ClientAndPtr<xla::PjRtDevice>& device, const std::string& kind) {
  xla::PjRtMemorySpace* result_memory_space = nullptr;
  for (auto* memory_space : device->memory_spaces()) {
    if (memory_space->memory_space_kind() == kind) {
      if (result_memory_space != nullptr) {
        std::string memories = absl::StrJoin(
            device->memory_spaces(), ", ",
            [](std::string* out, const auto& memory_space) {
              absl::StrAppend(out, memory_space->memory_space_kind());
            });
        auto device_kind = device->device_kind();
        xla::ThrowIfError(
            xla::InvalidArgument("Found more than one addressable memory for "
                                 "kind %s which is not allowed. There can only "
                                 "be one memory for each "
                                 "kind. Device %s can address the following "
                                 "memory kinds: %s",
                                 kind, device_kind, memories));
      }
      result_memory_space = memory_space;
    }
  }
  if (result_memory_space == nullptr) {
    std::string memories =
        absl::StrJoin(device->memory_spaces(), ", ",
                      [](std::string* out, const auto& memory_space) {
                        absl::StrAppend(out, memory_space->memory_space_kind());
                      });
    auto device_kind = device->device_kind();
    xla::ThrowIfError(xla::InvalidArgument(
        "Could not find memory addressable by device %s. Device %s "
        "can address the following memory kinds: %s. "
        "Got memory kind: %s",
        device_kind, device_kind, memories, kind));
  }
  return WrapWithClient(device.client(), result_memory_space);
}

NamedSharding::NamedSharding(py::object mesh, py::object spec,
                             py::object memory_kind, py::object parsed_pspec)
    : XLACompatibleSharding(/*num_devices=*/[&mesh]() {
        py::array devices = mesh.attr("devices");
        return devices.size();
      }()),
      mesh_(std::move(mesh)),
      spec_(std::move(spec)),
      memory_kind_(std::move(memory_kind)),
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
      .def(py::init<py::object, py::object, py::object, py::object>(),
           py::arg("mesh"), py::arg("spec"), py::kw_only(),
           py::arg("memory_kind") = py::none(),
           py::arg("_parsed_pspec") = py::none())
      .def_property_readonly("mesh", &NamedSharding::mesh)
      .def_property_readonly("spec", &NamedSharding::spec)
      .def_property_readonly("memory_kind", &NamedSharding::memory_kind)
      .def_property("_parsed_pspec", &NamedSharding::parsed_pspec,
                    &NamedSharding::set_parsed_pspec);

  py::class_<SingleDeviceSharding, XLACompatibleSharding>(
      m, "SingleDeviceSharding", py::dynamic_attr())
      .def(py::init<py::object, py::object>(), py::arg("device"), py::kw_only(),
           py::arg("memory_kind") = py::none())
      .def_property_readonly("_device", &SingleDeviceSharding::device)
      .def_property_readonly("_memory_kind",
                             &SingleDeviceSharding::memory_kind);

  py::class_<PmapSharding, XLACompatibleSharding>(m, "PmapSharding",
                                                  py::dynamic_attr())
      .def(py::init<py::object, ShardingSpec>(), py::arg("devices"),
           py::arg("sharding_spec"))
      .def_property_readonly("devices", &PmapSharding::devices)
      .def_property_readonly("sharding_spec", &PmapSharding::sharding_spec);

  py::class_<GSPMDSharding, XLACompatibleSharding>(m, "GSPMDSharding",
                                                   py::dynamic_attr())
      .def(py::init<py::list, xla::OpSharding, py::object>(),
           py::arg("devices"), py::arg("op_sharding"), py::kw_only(),
           py::arg("memory_kind") = py::none())
      .def(py::init<py::tuple, xla::OpSharding, py::object>(),
           py::arg("devices"), py::arg("op_sharding"), py::kw_only(),
           py::arg("memory_kind") = py::none())
      .def(py::init<py::list, xla::HloSharding, py::object>(),
           py::arg("devices"), py::arg("op_sharding"), py::kw_only(),
           py::arg("memory_kind") = py::none())
      .def(py::init<py::tuple, xla::HloSharding, py::object>(),
           py::arg("devices"), py::arg("op_sharding"), py::kw_only(),
           py::arg("memory_kind") = py::none())
      .def_property_readonly("_devices", &GSPMDSharding::devices)
      .def_property_readonly("_hlo_sharding", &GSPMDSharding::hlo_sharding)
      .def_property_readonly("_memory_kind", &GSPMDSharding::memory_kind);
}

}  // namespace jax
