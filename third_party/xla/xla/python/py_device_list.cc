/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/py_device_list.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/make_iterator.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_helpers.h"
#include "xla/python/py_client.h"
#include "xla/python/py_device.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/sharding.h"
#include "xla/python/types.h"
#include "xla/util.h"

namespace jax {

namespace nb = ::nanobind;

PyDeviceList::PyDeviceList(xla::nb_class_ptr<xla::PyClient> py_client,
                           xla::ifrt::DeviceList device_list)
    : py_client_(std::move(py_client)), device_list_(std::move(device_list)) {}

PyDeviceList::PyDeviceList(nb::tuple py_device_assignment)
    : device_list_(py_device_assignment) {
  // Attempt to convert to Python devices into `ifrt::DeviceList`.
  if (py_device_assignment.size() == 0) {
    device_list_ = xla::ifrt::DeviceList();
    return;
  }
  xla::ifrt::DeviceList::Devices devices;
  devices.reserve(py_device_assignment.size());
  for (nb::handle obj : py_device_assignment) {
    if (!nb::isinstance<xla::PyDevice>(obj.ptr())) {
      // Non-`xla::PyDevice` is used on an alternative JAX backend with device
      // duck typing. Use Python device objects already set in `device_list_`.
      return;
    }
    auto py_device = nb::cast<xla::PyDevice*>(obj);
    if (py_client_.get() == nullptr) {
      py_client_ = py_device->client();
    } else if (py_device->client().get() != py_client_.get()) {
      // If the list contains multiple clients, fall back to device duck typing.
      return;
    }
    devices.push_back(py_device->device());
  }
  device_list_ = xla::ifrt::DeviceList(std::move(devices));
}

PyDeviceList::~PyDeviceList() {
  if (device_list_.index() == 1) {
    xla::GlobalPyRefManager()->AddGarbage(
        std::move(std::get<1>(std::move(device_list_))));
  }
}

absl::StatusOr<xla::ifrt::DeviceList> PyDeviceList::ifrt_device_list() const {
  switch (device_list_.index()) {
    case 0:
      return std::get<0>(device_list_);
    case 1:
      return xla::InvalidArgument("DeviceList contains non-IFRT devices");
    default:
      return xla::InvalidArgument("Unrecognized DeviceList type");
  }
}

int64_t PyDeviceList::Hash() {
  if (!hash_.has_value()) {
    switch (device_list_.index()) {
      case 0:
        hash_ = absl::HashOf(std::get<0>(device_list_));
        break;
      case 1:
        hash_ = xla::nb_hash(std::get<1>(device_list_));
        break;
      default:
        throw nb::value_error("Unrecognized DeviceList type");
    }
  }
  return *hash_;
}

bool PyDeviceList::operator==(nb::handle other) {
  if (!nb::isinstance<PyDeviceList>(other)) {
    return false;
  }
  auto o = nb::cast<PyDeviceList*>(other);
  // Fast-path using a pointer equality check.
  if (this == o) {
    return true;
  }
  if (Hash() != o->Hash()) {
    return false;
  }
  if (device_list_.index() == 0 && o->device_list_.index() == 0) {
    nb::gil_scoped_release gil_release;
    return std::get<0>(device_list_) == std::get<0>(o->device_list_);
  } else {
    return AsTuple().equal(o->AsTuple());
  }
}

bool PyDeviceList::operator!=(nb::handle other) { return !(*this == other); }

int PyDeviceList::Len() const {
  switch (device_list_.index()) {
    case 0:
      return std::get<0>(device_list_).size();
    case 1:
      return nb::len(std::get<1>(device_list_));
    default:
      throw nb::value_error("Unrecognized DeviceList type");
  }
}

nb::object PyDeviceList::GetItem(int index) {
  switch (device_list_.index()) {
    case 0: {
      const xla::ifrt::DeviceList& device_list = std::get<0>(device_list_);
      if (index < -device_list.size() || index >= device_list.size()) {
        throw nb::index_error();
      } else if (index < 0) {
        index += device_list.size();
      }
      return py_client_->GetPyDevice(device_list[index]);
    }
    case 1:
      return std::get<1>(device_list_).attr("__getitem__")(index);
    default:
      throw nb::value_error("Unrecognized DeviceList type");
  }
}

nb::object PyDeviceList::GetSlice(nb::slice slice) {
  switch (device_list_.index()) {
    case 0: {
      const xla::ifrt::DeviceList& device_list = std::get<0>(device_list_);
      Py_ssize_t start, stop, step, slicelength;
      if (PySlice_GetIndicesEx(slice.ptr(), device_list.size(), &start, &stop,
                               &step, &slicelength) != 0) {
        throw nb::python_error();
      }
      nb::tuple out = nb::steal<nb::tuple>(PyTuple_New(slicelength));
      for (size_t i = 0; i < slicelength; ++i) {
        nb::object d = py_client_->GetPyDevice(device_list[start]);
        PyTuple_SET_ITEM(out.ptr(), i, d.release().ptr());
        start += step;
      }
      return std::move(out);
    }
    case 1:
      return std::get<1>(device_list_).attr("__getitem__")(slice);
    default:
      throw nb::value_error("Unrecognized DeviceList type");
  }
}

nb::tuple PyDeviceList::AsTuple() const {
  switch (device_list_.index()) {
    case 0: {
      const xla::ifrt::DeviceList& device_list = std::get<0>(device_list_);
      nb::tuple out = nb::steal<nb::tuple>(PyTuple_New(device_list.size()));
      int i = 0;
      for (xla::ifrt::Device* device : device_list) {
        nb::object d = py_client_->GetPyDevice(device);
        PyTuple_SET_ITEM(out.ptr(), i, d.release().ptr());
        ++i;
      }
      return out;
    }
    case 1:
      return std::get<1>(device_list_);
    default:
      throw nb::value_error("Unrecognized DeviceList type");
  }
}

nb::iterator PyDeviceList::Iter() {
  switch (device_list_.index()) {
    case 0: {
      // Iterator whose deference converts `xla::ifrt::Device*` into JAX
      // `PjRtDevice`.
      struct Iterator {
        void operator++() { ++it; }
        bool operator==(const Iterator& other) const { return it == other.it; }
        xla::nb_class_ptr<xla::PyDevice> operator*() const {
          return py_client->GetPyDevice(*it);
        }
        xla::nb_class_ptr<xla::PyClient> py_client;
        xla::ifrt::DeviceList::Devices::const_iterator it;
      };
      return nb::make_iterator(
          nb::type<PyDeviceList>(), "ifrt_device_iterator",
          Iterator{py_client_, std::get<0>(device_list_).begin()},
          Iterator{py_client_, std::get<0>(device_list_).end()});
    }
    case 1:
      return nb::make_iterator(
          nb::type<PyDeviceList>(), "python_device_iterator",
          std::get<1>(device_list_).begin(), std::get<1>(device_list_).end());
    default:
      throw nb::value_error("Unrecognized DeviceList type");
  }
}

std::string PyDeviceList::Str() {
  return nb::cast<std::string>(nb::str(AsTuple()));
}

nb::tuple PyDeviceList::Dump() const { return AsTuple(); }

bool PyDeviceList::IsFullyAddressable() {
  if (!is_fully_addressable_.has_value()) {
    is_fully_addressable_ = true;
    switch (device_list_.index()) {
      case 0: {
        const int process_index = py_client_ ? py_client_->process_index() : 0;
        for (const xla::ifrt::Device* device :
             std::get<0>(device_list_).devices()) {
          if (device->process_index() != process_index) {
            is_fully_addressable_ = false;
            break;
          }
        }
        break;
      }
      case 1: {
        for (nb::handle device : std::get<1>(device_list_)) {
          if (nb::cast<int>(device.attr("process_index")) !=
              nb::cast<int>(device.attr("client").attr("process_index")())) {
            is_fully_addressable_ = false;
            break;
          }
        }
        break;
      }
      default:
        throw nb::value_error("Unrecognized DeviceList type");
    }
  }
  return *is_fully_addressable_;
}

/*static*/ xla::nb_class_ptr<PyDeviceList> PyDeviceList::AddressableDeviceList(
    xla::nb_class_ptr<PyDeviceList> self) {
  if (self->IsFullyAddressable()) {
    // Do not cache this result in `addressable_device_list_`. Otherwise, it
    // will create a cycle that prevents deletion of this object.
    return self;
  }
  if (!self->addressable_device_list_.has_value()) {
    switch (self->device_list_.index()) {
      case 0: {
        xla::ifrt::DeviceList::Devices addressable_devices;
        const int process_index =
            self->py_client_ ? self->py_client_->process_index() : 0;
        for (xla::ifrt::Device* device :
             std::get<0>(self->device_list_).devices()) {
          if (device->process_index() == process_index) {
            addressable_devices.push_back(device);
          }
        }
        self->addressable_device_list_ = xla::make_nb_class<PyDeviceList>(
            self->py_client_,
            xla::ifrt::DeviceList(std::move(addressable_devices)));
        break;
      }
      case 1: {
        auto device_list = std::get<1>(self->device_list_);
        std::vector<nb::object> addressable_devices;
        for (size_t i = 0; i < device_list.size(); ++i) {
          nb::object device = device_list[i];
          if (nb::cast<int>(device.attr("process_index")) ==
              nb::cast<int>(device.attr("client").attr("process_index")())) {
            addressable_devices.push_back(std::move(device));
          }
        }
        self->addressable_device_list_ = xla::make_nb_class<PyDeviceList>(
            xla::MutableSpanToNbTuple(absl::MakeSpan(addressable_devices)));
        break;
      }
      default:
        throw nb::value_error("Unrecognized DeviceList type");
    }
  }
  return *self->addressable_device_list_;
}

void PyDeviceList::PopulateMemoryKindInfo() {
  if (device_list_.index() == 1) {
    // Handle Python duck-type devices in a separate function for readability.
    PopulateMemoryKindInfoForDuckTypedDevices();
    return;
  }
  if (device_list_.index() != 0) {
    throw nb::value_error("Unrecognized DeviceList type");
  }
  MemoryKindInfo info;
  if (!GetEnableMemories()) {
    info.default_memory_kind = nb::none();
    memory_kind_info_ = std::move(info);
    return;
  }
  xla::ifrt::Device* addressable_device = nullptr;
  const int process_index = py_client_ ? py_client_->process_index() : 0;
  for (xla::ifrt::Device* device : std::get<0>(device_list_).devices()) {
    if (device->process_index() == process_index) {
      addressable_device = device;
      break;
    }
  }
  if (addressable_device == nullptr) {
    info.default_memory_kind = nb::none();
    memory_kind_info_ = std::move(info);
    return;
  }

  auto default_memory = addressable_device->default_memory_space();
  if (!default_memory.ok()) {
    // Cache the error.
    memory_kind_info_ = default_memory.status();
    return;
  }
  info.default_memory_kind = nb::cast(std::string((*default_memory)->kind()));
  nb::tuple memory_kinds = nb::steal<nb::tuple>(
      PyTuple_New(addressable_device->memory_spaces().size()));
  for (size_t i = 0; i < addressable_device->memory_spaces().size(); ++i) {
    auto* memory = addressable_device->memory_spaces()[i];
    nb::str s = nb::str(memory->kind().data(), memory->kind().size());
    PyTuple_SET_ITEM(memory_kinds.ptr(), i, s.release().ptr());
  }
  info.memory_kinds = std::move(memory_kinds);
  memory_kind_info_ = std::move(info);
}

void PyDeviceList::PopulateMemoryKindInfoForDuckTypedDevices() {
  MemoryKindInfo info;
  if (!GetEnableMemories()) {
    info.default_memory_kind = nb::none();
    // info.memory_kinds is default-initialized to an empty tuple.
    memory_kind_info_ = std::move(info);
    return;
  }
  try {
    nb::handle addressable_device;
    for (nb::handle device : std::get<1>(device_list_)) {
      if (nb::cast<int>(device.attr("process_index")) ==
          nb::cast<int>(device.attr("client").attr("process_index")())) {
        addressable_device = device;
        break;
      }
    }
    if (!addressable_device) {
      info.default_memory_kind = nb::none();
      // info.memory_kinds is default-initialized to an empty tuple.
      memory_kind_info_ = std::move(info);
      return;
    }
    auto default_memory = addressable_device.attr("default_memory")();
    info.default_memory_kind = default_memory.attr("kind");
    info.memory_kinds = nb::tuple(
        nb::object(addressable_device.attr("addressable_memories")()));
    memory_kind_info_ = std::move(info);
  } catch (nb::python_error& e) {
    // Cache the error.
    memory_kind_info_ = xla::InvalidArgument("%s", e.what());
  }
}

absl::StatusOr<nb::tuple> PyDeviceList::MemoryKinds() {
  if (!memory_kind_info_.has_value()) {
    PopulateMemoryKindInfo();
  }
  if (!memory_kind_info_->ok()) {
    return memory_kind_info_->status();
  }
  return (*memory_kind_info_)->memory_kinds;
}

absl::StatusOr<nb::object> PyDeviceList::DefaultMemoryKind() {
  if (!memory_kind_info_.has_value()) {
    PopulateMemoryKindInfo();
  }
  if (!memory_kind_info_->ok()) {
    return memory_kind_info_->status();
  }
  return (*memory_kind_info_)->default_memory_kind;
}

void RegisterDeviceList(nb::module_& m) {
  nb::class_<PyDeviceList>(m, "DeviceList")
      .def(nb::init<nb::tuple>())
      .def("__hash__", &PyDeviceList::Hash)
      .def("__eq__", &PyDeviceList::operator==)
      .def("__ne__", &PyDeviceList::operator!=)
      .def("__len__", &PyDeviceList::Len)
      .def("__getitem__", &PyDeviceList::GetItem)
      .def("__getitem__", &PyDeviceList::GetSlice)
      .def("__iter__", &PyDeviceList::Iter, nb::keep_alive<0, 1>())
      .def("__str__", &PyDeviceList::Str)
      .def("__repr__", &PyDeviceList::Str)
      .def("__getstate__", [](const PyDeviceList& l) { return l.Dump(); })
      .def("__setstate__",
           [](PyDeviceList& self, nb::tuple t) {
             new (&self) PyDeviceList(std::move(t));
           })
      .def_prop_ro("is_fully_addressable", &PyDeviceList::IsFullyAddressable)
      .def_prop_ro("addressable_device_list",
                   &PyDeviceList::AddressableDeviceList)
      // `xla::ValueOrThrowWrapper` does not work with
      // `def_prop_ro()`. Manually convert an error into an exception.
      .def_prop_ro("default_memory_kind",
                   [](PyDeviceList* l) {
                     auto kind = l->DefaultMemoryKind();
                     if (!kind.ok()) {
                       throw nb::value_error(kind.status().ToString().c_str());
                     }
                     return *kind;
                   })
      .def_prop_ro("memory_kinds", [](PyDeviceList* l) {
        auto kinds = l->MemoryKinds();
        if (!kinds.ok()) {
          throw nb::value_error(kinds.status().ToString().c_str());
        }
        return *kinds;
      });
}

}  // namespace jax
