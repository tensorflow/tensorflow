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
#include <optional>
#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_client.h"
#include "xla/python/py_device_list.h"
#include "xla/python/sharded_device_array.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace jax {

namespace nb = nanobind;

// Gets `jax::PyDeviceList` from a JAX Sharding.
absl::StatusOr<xla::nb_class_ptr<jax::PyDeviceList>> GetPyDeviceList(
    nb::handle sharding_py) {
  nb::handle sharding(sharding_py.ptr());
  if (sharding.type().is(jax::NamedSharding::type())) {
    TF_ASSIGN_OR_RETURN(
        auto ns_device_list,
        nb::cast<const jax::NamedSharding*>(sharding)->internal_device_list());
    return ns_device_list;
  } else if (sharding.type().is(jax::SingleDeviceSharding::type())) {
    return nb::cast<const jax::SingleDeviceSharding*>(sharding)
        ->internal_device_list();
  } else if (sharding.type().is(jax::PmapSharding::type())) {
    return nb::cast<const jax::PmapSharding*>(sharding)->internal_device_list();
  } else if (sharding.type().is(jax::GSPMDSharding::type())) {
    return nb::cast<const jax::GSPMDSharding*>(sharding)
        ->internal_device_list();
  } else {
    return nb::cast<xla::nb_class_ptr<jax::PyDeviceList>>(
        sharding.attr("_internal_device_list"));
  }
}

nb::object CheckAndCanonicalizeMemoryKind(
    nb::object memory_kind,
    const xla::nb_class_ptr<PyDeviceList>& device_list) {
  if (!memory_kind.is_none()) {
    // If memory kind is not None, check if it's supported by the devices
    // mentioned in the Sharding.
    auto supported_memory_kinds = PyDeviceList::MemoryKinds(device_list);
    if (!supported_memory_kinds.ok()) {
      supported_memory_kinds = nb::tuple();
    }
    for (nb::handle supported_memory_kind : *supported_memory_kinds) {
      if (supported_memory_kind.equal(memory_kind)) {
        return memory_kind;
      }
    }
    auto addressable_device_list =
        PyDeviceList::AddressableDeviceList(device_list);
    if (addressable_device_list->Len() == 0) {
      // If the device list is not addressable, we can't check if the memory
      // kind is supported, so we assume it is.
      return memory_kind;
    }
    nb::object device_kind =
        addressable_device_list->GetItem(0).attr("device_kind");
    absl::string_view device_kind_str =
        nb::cast<absl::string_view>(device_kind);
    auto py_str_formatter = [](std::string* out, nb::handle h) {
      *out += nb::cast<absl::string_view>(nb::str(h));
    };
    throw nb::value_error(
        absl::StrCat(
            "Could not find memory addressable by device ", device_kind_str,
            ". Device ", device_kind_str,
            " can address the following memory kinds: ",
            absl::StrJoin(*supported_memory_kinds, ", ", py_str_formatter),
            ". Got memory kind: ", nb::cast<absl::string_view>(memory_kind))
            .c_str());
  }
  // If memory kind is None, canonicalize to default memory.
  absl::StatusOr<nb::object> default_memory_kind =
      PyDeviceList::DefaultMemoryKind(device_list);
  if (!default_memory_kind.ok()) {
    return nb::none();
  }
  return *std::move(default_memory_kind);
}

int Sharding::SafeNumDevices(nb::handle sharding) {
  const jax::Sharding* cpp_sharding;
  if (nb::try_cast<const jax::Sharding*>(sharding, cpp_sharding)) {
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

  return nb::hash(sharding);
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
               b_named_sharding->manual_axes()) &&
           a_named_sharding->logical_device_ids().equal(
               b_named_sharding->logical_device_ids());
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
                             nb::object memory_kind, nb::object manual_axes,
                             nb::object logical_device_ids)
    : Sharding(/*num_devices=*/[&mesh]() {
        return nb::cast<int>(mesh.attr("size"));
      }()),
      mesh_(std::move(mesh)),
      spec_(std::move(spec)),
      memory_kind_(std::move(memory_kind)),
      manual_axes_(std::move(manual_axes)),
      logical_device_ids_(std::move(logical_device_ids)) {
  if (spec_.is_none()) {
    throw nb::type_error(
        "Unexpected None passed as spec for NamedSharding. Did you mean P()?");
  }
  nb::object idl = nb::object(mesh_.attr("_internal_device_list"));
  if (idl.is_none()) {
    internal_device_list_ = std::nullopt;
  } else {
    internal_device_list_ = nb::cast<xla::nb_class_ptr<jax::PyDeviceList>>(idl);
  }
  if (internal_device_list_) {
    memory_kind_ =
        CheckAndCanonicalizeMemoryKind(memory_kind_, *internal_device_list_);
  } else {
    memory_kind_ = nb::none();
  }

  // TODO(phawkins): this leaks a reference to the check_pspec function.
  // A better way to fix this would be to move PartitionSpec and this check into
  // C++.
  static nb::object* check_pspec = []() {
    nb::module_ si = nb::module_::import_("jax._src.named_sharding");
    return new nb::object(si.attr("check_pspec"));
  }();
  (*check_pspec)(mesh_, spec_, manual_axes_);
}

SingleDeviceSharding::SingleDeviceSharding(nb::object device,
                                           nb::object memory_kind)
    : Sharding(/*num_devices=*/1),
      device_(device),
      memory_kind_(std::move(memory_kind)),
      internal_device_list_(
          xla::make_nb_class<PyDeviceList>(nb::make_tuple(std::move(device)))) {
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

SingleDeviceSharding::SingleDeviceSharding(
    xla::nb_class_ptr<xla::PyClient> client,
    xla::ifrt::DeviceListRef device_list, nb::object memory_kind)
    : Sharding(/*num_devices=*/1),
      device_(client->GetPyDevice(device_list->devices().front())),
      memory_kind_(std::move(memory_kind)),
      internal_device_list_(xla::make_nb_class<PyDeviceList>(
          std::move(client), std::move(device_list))) {
  memory_kind_ =
      CheckAndCanonicalizeMemoryKind(memory_kind_, internal_device_list_);
}

PmapSharding::PmapSharding(xla::nb_numpy_ndarray devices,
                           ShardingSpec sharding_spec)
    : Sharding(/*num_devices=*/devices.size()),
      devices_(std::move(devices)),
      sharding_spec_(std::move(sharding_spec)) {
  nb::object flat_devices = devices_.attr("flat");
  internal_device_list_ =
      xla::make_nb_class<PyDeviceList>(nb::tuple(flat_devices));
}

GSPMDSharding::GSPMDSharding(nb::sequence devices, xla::HloSharding op_sharding,
                             nb::object memory_kind, nb::object device_list)
    : Sharding(/*num_devices=*/nb::len(devices.ptr())),
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

  nb::class_<NamedSharding, Sharding>(m, "NamedSharding", nb::dynamic_attr())
      .def(nb::init<nb::object, nb::object, nb::object, nb::object,
                    nb::object>(),
           nb::arg("mesh"), nb::arg("spec").none(),
           nb::arg("memory_kind").none() = nb::none(),
           nb::arg("_manual_axes") = nb::steal(PyFrozenSet_New(nullptr)),
           nb::arg("_logical_device_ids").none() = nb::none())
      .def_prop_ro("mesh", &NamedSharding::mesh)
      .def_prop_ro("spec", &NamedSharding::spec)
      .def_prop_ro("_memory_kind", &NamedSharding::memory_kind)
      .def_prop_ro("_manual_axes", &NamedSharding::manual_axes)
      .def_prop_ro("_logical_device_ids", &NamedSharding::logical_device_ids)
      .def_prop_ro("_internal_device_list", [](const NamedSharding& s) {
        return xla::ValueOrThrow(s.internal_device_list());
      });

  nb::class_<SingleDeviceSharding, Sharding>(m, "SingleDeviceSharding",
                                             nb::dynamic_attr())
      .def(nb::init<nb::object, nb::object>(), nb::arg("device"),
           nb::arg("memory_kind").none() = nb::none())
      .def_prop_ro("_device", &SingleDeviceSharding::device)
      .def_prop_ro("_memory_kind", &SingleDeviceSharding::memory_kind)
      .def_prop_ro("_internal_device_list",
                   &SingleDeviceSharding::internal_device_list);

  nb::class_<PmapSharding, Sharding>(m, "PmapSharding", nb::dynamic_attr())
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

  nb::class_<GSPMDSharding, Sharding>(m, "GSPMDSharding", nb::dynamic_attr())
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
