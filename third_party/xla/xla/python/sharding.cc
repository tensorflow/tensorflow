/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/sharding.h"

#include <Python.h>

#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_helpers.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_client.h"
#include "xla/python/py_device_list.h"
#include "xla/python/sharded_device_array.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace jax {

namespace nb = nanobind;

bool (*GetEnableMemories)() = +[] {
  static bool fetch_memory_kind_on_executable = [] {
    char* v = getenv("JAX_ENABLE_MEMORIES");
    if (v == nullptr || *v == '\0') {
      return false;
    }
    return true;
  }();
  return fetch_memory_kind_on_executable;
};

nb::object CheckAndCanonicalizeMemoryKind(
    nb::object memory_kind,
    const xla::nb_class_ptr<PyDeviceList>& device_list) {
  if (!memory_kind.is_none()) {
    // If memory kind is not None, check if it's supported by the devices
    // mentioned in the Sharding.
    auto supported_memory_kinds = device_list->MemoryKinds();
    if (!supported_memory_kinds.ok()) {
      supported_memory_kinds = nb::tuple();
    }
    for (nb::handle supported_memory_kind : *supported_memory_kinds) {
      if (supported_memory_kind.equal(memory_kind)) {
        return memory_kind;
      }
    }
    nb::object device_kind = PyDeviceList::AddressableDeviceList(device_list)
                                 ->GetItem(0)
                                 .attr("device_kind");
    std::string_view device_kind_str = nb::cast<std::string_view>(device_kind);
    auto py_str_formatter = [](std::string* out, nb::handle h) {
      *out += nb::cast<std::string_view>(nb::str(h));
    };
    throw nb::value_error(
        absl::StrCat(
            "Could not find memory addressable by device ", device_kind_str,
            ". Device ", device_kind_str,
            " can address the following memory kinds: ",
            absl::StrJoin(*supported_memory_kinds, ", ", py_str_formatter),
            ". Got memory kind: ", nb::cast<std::string_view>(memory_kind))
            .c_str());
  }
  // If memory kind is None, canonicalize to default memory.
  absl::StatusOr<nb::object> default_memory_kind =
      device_list->DefaultMemoryKind();
  if (!default_memory_kind.ok()) {
    return nb::none();
  }
  return *std::move(default_memory_kind);
}

int Sharding::SafeNumDevices(nb::handle sharding) {
  // Pure python shardings are not initialized, so we should not
  // even be casting if they are not initialized.
  if (nb::inst_check(sharding) && nb::inst_ready(sharding)) {
    const auto* cpp_sharding = nb::cast<const jax::Sharding*>(sharding);
    if (cpp_sharding->num_devices_.has_value()) {
      return (*cpp_sharding->num_devices_);
    }
  }

  nb::set device_set = sharding.attr("device_set");
  return device_set.size();
}

size_t ShardingHash(nb::handle sharding) {
  auto type = sharding.type();

  if (type.is(NamedSharding::type())) {
    const auto* named_sharding = nb::inst_ptr<jax::NamedSharding>(sharding);
    return absl::Hash<void*>()(named_sharding->mesh().ptr());
  }

  if (type.is(GSPMDSharding::type())) {
    auto* gspmd_sharding = nb::inst_ptr<GSPMDSharding>(sharding);
    return gspmd_sharding->Hash();
  }

  if (type.is(SingleDeviceSharding::type())) {
    auto* single_device_sharding = nb::inst_ptr<SingleDeviceSharding>(sharding);
    return absl::Hash<void*>()(single_device_sharding->device().ptr());
  }

  return xla::nb_hash(sharding);
}

bool ShardingEqual(nb::handle a, nb::handle b) {
  if (a.ptr() == b.ptr()) return true;

  auto a_type = a.type();
  auto b_type = b.type();

  if (!a_type.is(b_type)) return false;

  if (a_type.is(NamedSharding::type())) {
    auto* a_named_sharding = nb::inst_ptr<const NamedSharding>(a);
    auto* b_named_sharding = nb::inst_ptr<const NamedSharding>(b);

    return a_named_sharding->mesh().ptr() == b_named_sharding->mesh().ptr() &&
           a_named_sharding->spec().equal(b_named_sharding->spec()) &&
           a_named_sharding->memory_kind().equal(
               b_named_sharding->memory_kind()) &&
           a_named_sharding->manual_axes().equal(
               b_named_sharding->manual_axes());
  }

  if (a_type.is(GSPMDSharding::type())) {
    auto* a_gspmd_sharding = nb::inst_ptr<const GSPMDSharding>(a);
    auto* b_gspmd_sharding = nb::inst_ptr<const GSPMDSharding>(b);

    return a_gspmd_sharding == b_gspmd_sharding;
  }

  if (a_type.is(SingleDeviceSharding::type())) {
    auto* a_single_device_sharding =
        nb::inst_ptr<const SingleDeviceSharding>(a);
    auto* b_single_device_sharding =
        nb::inst_ptr<const SingleDeviceSharding>(b);

    return a_single_device_sharding->device().ptr() ==
               b_single_device_sharding->device().ptr() &&
           a_single_device_sharding->memory_kind().equal(
               b_single_device_sharding->memory_kind());
  }

  return a.equal(b);
}

NamedSharding::NamedSharding(nb::object mesh, nb::object spec,
                             nb::object memory_kind, nb::object parsed_pspec,
                             nb::object manual_axes)
    : XLACompatibleSharding(/*num_devices=*/[&mesh]() {
        xla::nb_numpy_ndarray devices = mesh.attr("devices");
        return devices.size();
      }()),
      mesh_(std::move(mesh)),
      spec_(std::move(spec)),
      memory_kind_(std::move(memory_kind)),
      parsed_pspec_(std::move(parsed_pspec)),
      manual_axes_(std::move(manual_axes)) {
  nb::object idl = nb::object(mesh_.attr("_internal_device_list"));
  internal_device_list_ = nb::cast<xla::nb_class_ptr<jax::PyDeviceList>>(
      nb::object(mesh_.attr("_internal_device_list")));
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);

  nb::module_ si = nb::module_::import_("jax._src.sharding_impls");
  parsed_pspec_ = si.attr("preprocess")(mesh_, spec_, parsed_pspec_);
}

SingleDeviceSharding::SingleDeviceSharding(nb::object device,
                                           nb::object memory_kind)
    : XLACompatibleSharding(/*num_devices=*/1),
      device_(device),
      memory_kind_(std::move(memory_kind)),
      internal_device_list_(
          xla::make_nb_class<PyDeviceList>(nb::make_tuple(std::move(device)))) {
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

SingleDeviceSharding::SingleDeviceSharding(
    xla::nb_class_ptr<xla::PyClient> client, xla::ifrt::DeviceList device_list,
    nb::object memory_kind)
    : XLACompatibleSharding(/*num_devices=*/1),
      device_(client->GetPyDevice(device_list.front())),
      memory_kind_(std::move(memory_kind)),
      internal_device_list_(xla::make_nb_class<PyDeviceList>(
          std::move(client), std::move(device_list))) {
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

PmapSharding::PmapSharding(xla::nb_numpy_ndarray devices,
                           ShardingSpec sharding_spec)
    : XLACompatibleSharding(/*num_devices=*/devices.size()),
      devices_(std::move(devices)),
      sharding_spec_(std::move(sharding_spec)) {
  nb::object flat_devices = devices_.attr("flat");
  internal_device_list_ =
      xla::make_nb_class<PyDeviceList>(nb::tuple(flat_devices));
}

GSPMDSharding::GSPMDSharding(nb::sequence devices, xla::HloSharding op_sharding,
                             nb::object memory_kind, nb::object device_list)
    : XLACompatibleSharding(/*num_devices=*/nb::len(devices.ptr())),
      devices_(nb::tuple(devices)),
      hlo_sharding_(std::move(op_sharding)),
      memory_kind_(std::move(memory_kind)) {
  if (device_list.is_none()) {
    internal_device_list_ = xla::make_nb_class<PyDeviceList>(devices_);
  } else {
    internal_device_list_ =
        nb::cast<xla::nb_class_ptr<jax::PyDeviceList>>(std::move(device_list));
  }
  // This checks in python if the memory kind is correct for the given
  // devices. Currently in python this check is optimized but we want to
  // move that check to C++ after which we can remove this call.
  CHECK(devices_.size() != 0)
      << "Devices given to GSPMDSharding must not be empty";
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

void RegisterSharding(nb::module_& m) {
  nb::class_<Sharding>(m, "Sharding").def(nb::init<>());

  nb::class_<XLACompatibleSharding, Sharding>(m, "XLACompatibleSharding")
      .def(nb::init<>());

  nb::class_<NamedSharding, XLACompatibleSharding>(m, "NamedSharding",
                                                   nb::dynamic_attr())
      .def(nb::init<nb::object, nb::object, nb::object, nb::object,
                    nb::object>(),
           nb::arg("mesh"), nb::arg("spec").none(),
           nb::arg("memory_kind").none() = nb::none(),
           nb::arg("_parsed_pspec").none() = nb::none(),
           nb::arg("_manual_axes") = nb::steal(PyFrozenSet_New(nullptr)))
      .def_prop_ro("mesh", &NamedSharding::mesh)
      .def_prop_ro("spec", &NamedSharding::spec)
      .def_prop_ro("_memory_kind", &NamedSharding::memory_kind)
      .def_prop_ro("_manual_axes", &NamedSharding::manual_axes)
      .def_prop_rw("_parsed_pspec", &NamedSharding::parsed_pspec,
                   &NamedSharding::set_parsed_pspec)
      .def_prop_ro("_internal_device_list",
                   &NamedSharding::internal_device_list);

  nb::class_<SingleDeviceSharding, XLACompatibleSharding>(
      m, "SingleDeviceSharding", nb::dynamic_attr())
      .def(nb::init<nb::object, nb::object>(), nb::arg("device"),
           nb::arg("memory_kind").none() = nb::none())
      .def_prop_ro("_device", &SingleDeviceSharding::device)
      .def_prop_ro("_memory_kind", &SingleDeviceSharding::memory_kind)
      .def_prop_ro("_internal_device_list",
                   &SingleDeviceSharding::internal_device_list);

  nb::class_<PmapSharding, XLACompatibleSharding>(m, "PmapSharding",
                                                  nb::dynamic_attr())
      .def(
          "__init__",
          [](PmapSharding* self, nb::object devices,
             ShardingSpec sharding_spec) {
            new (self) PmapSharding(xla::nb_numpy_ndarray::ensure(devices),
                                    std::move(sharding_spec));
          },
          nb::arg("devices"), nb::arg("sharding_spec"))
      .def_prop_ro("devices", &PmapSharding::devices)
      .def_prop_ro("sharding_spec", &PmapSharding::sharding_spec)
      .def_prop_ro("_internal_device_list",
                   &PmapSharding::internal_device_list);

  nb::class_<GSPMDSharding, XLACompatibleSharding>(m, "GSPMDSharding",
                                                   nb::dynamic_attr())
      .def(nb::init<nb::sequence, xla::OpSharding, nb::object, nb::object>(),
           nb::arg("devices"), nb::arg("op_sharding"),
           nb::arg("memory_kind").none() = nb::none(),
           nb::arg("_device_list").none() = nb::none())
      .def(nb::init<nb::sequence, xla::HloSharding, nb::object, nb::object>(),
           nb::arg("devices"), nb::arg("op_sharding"),
           nb::arg("memory_kind").none() = nb::none(),
           nb::arg("_device_list").none() = nb::none())
      .def_prop_ro("_devices", &GSPMDSharding::devices)
      .def_prop_ro("_hlo_sharding", &GSPMDSharding::hlo_sharding)
      .def_prop_ro("_memory_kind", &GSPMDSharding::memory_kind)
      .def_prop_ro("_internal_device_list",
                   &GSPMDSharding::internal_device_list);
}

}  // namespace jax
