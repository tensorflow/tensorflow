/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/types/span.h"
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11  // NOLINT
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/py_client.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/sharding.h"
#include "xla/statusor.h"
#include "xla/util.h"

namespace jax {

namespace py = ::pybind11;

PyDeviceList::PyDeviceList(std::shared_ptr<xla::PyClient> py_client,
                           xla::ifrt::DeviceList device_list)
    : py_client_(std::move(py_client)), device_list_(std::move(device_list)) {}

PyDeviceList::PyDeviceList(py::tuple py_device_assignment)
    : device_list_(py_device_assignment) {
  // Attempt to convert to Python devices into `ifrt::DeviceList`.
  if (py_device_assignment.empty()) {
    device_list_ = xla::ifrt::DeviceList({});
    return;
  }
  xla::ifrt::DeviceList::Devices devices;
  devices.reserve(devices.size());
  for (py::handle obj : py_device_assignment) {
    if (!py::isinstance<xla::PjRtDevice>(obj)) {
      // Non-`xla::PjRtDevice` is used on an alternative JAX backend with device
      // duck typing. Use Python device objects already set in `device_list_`.
      return;
    }
    auto py_device = py::cast<xla::ClientAndPtr<xla::PjRtDevice>>(obj);
    if (py_client_ == nullptr) {
      py_client_ = py_device.client();
    } else if (py_device.client() != py_client_) {
      // If the list contains multiple clients, fall back to device duck typing.
      return;
    }
    devices.push_back(py_device.get());
  }
  device_list_ = xla::ifrt::DeviceList(std::move(devices));
}

PyDeviceList::~PyDeviceList() {
  if (device_list_.index() == 1) {
    py::object py_device_assignment =
        py::cast<py::object>(std::get<1>(std::move(device_list_)));
    xla::GlobalPyRefManager()->AddGarbage(
        absl::MakeSpan(&py_device_assignment, 1));
  }
}

xla::StatusOr<xla::ifrt::DeviceList> PyDeviceList::ifrt_device_list() const {
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
        hash_ = py::hash(std::get<1>(device_list_));
        break;
      default:
        throw py::value_error("Unrecognized DeviceList type");
    }
  }
  return *hash_;
}

bool PyDeviceList::operator==(py::handle other) {
  if (!py::isinstance<PyDeviceList>(other)) {
    return false;
  }
  auto o = py::cast<std::shared_ptr<PyDeviceList>>(other);
  // Fast-path using a pointer equality check.
  if (this == o.get()) {
    return true;
  }
  if (Hash() != o->Hash()) {
    return false;
  }
  if (device_list_.index() == 0 && o->device_list_.index() == 0) {
    py::gil_scoped_release gil_release;
    return std::get<0>(device_list_) == std::get<0>(o->device_list_);
  } else {
    return AsTuple().equal(o->AsTuple());
  }
}

bool PyDeviceList::operator!=(py::handle other) { return !(*this == other); }

int PyDeviceList::Len() const {
  switch (device_list_.index()) {
    case 0:
      return std::get<0>(device_list_).size();
    case 1:
      return py::len(std::get<1>(device_list_));
    default:
      throw py::value_error("Unrecognized DeviceList type");
  }
}

py::object PyDeviceList::GetItem(int index) {
  switch (device_list_.index()) {
    case 0: {
      const xla::ifrt::DeviceList& device_list = std::get<0>(device_list_);
      if (index < -device_list.size() || index >= device_list.size()) {
        throw py::index_error();
      } else if (index < 0) {
        index += device_list.size();
      }
      return py::cast(xla::WrapWithClient(py_client_, device_list[index]));
    }
    case 1:
      return std::get<1>(device_list_).attr("__getitem__")(index);
    default:
      throw py::value_error("Unrecognized DeviceList type");
  }
}

py::object PyDeviceList::GetSlice(py::slice slice) {
  switch (device_list_.index()) {
    case 0: {
      const xla::ifrt::DeviceList& device_list = std::get<0>(device_list_);
      size_t start, stop, step, slicelength;
      if (!slice.compute(device_list.size(), &start, &stop, &step,
                         &slicelength)) {
        throw py::error_already_set();
      }
      std::vector<xla::ClientAndPtr<xla::PjRtDevice>> out;
      out.reserve(slicelength);
      for (size_t i = 0; i < slicelength; ++i) {
        out.push_back(xla::WrapWithClient(py_client_, device_list[start]));
        start += step;
      }
      return py::cast(out);
    }
    case 1:
      return std::get<1>(device_list_).attr("__getitem__")(slice);
    default:
      throw py::value_error("Unrecognized DeviceList type");
  }
}

py::tuple PyDeviceList::AsTuple() {
  switch (device_list_.index()) {
    case 0: {
      const xla::ifrt::DeviceList& device_list = std::get<0>(device_list_);
      std::vector<xla::ClientAndPtr<xla::PjRtDevice>> out;
      out.reserve(device_list.size());
      for (xla::ifrt::Device* device : device_list) {
        out.push_back(xla::WrapWithClient(py_client_, device));
      }
      return py::cast(out);
    }
    case 1:
      return std::get<1>(device_list_);
    default:
      throw py::value_error("Unrecognized DeviceList type");
  }
}

py::iterator PyDeviceList::Iter() {
  switch (device_list_.index()) {
    case 0: {
      // Iterator whose deference converts `xla::ifrt::Device*` into JAX
      // `PjRtDevice`.
      struct Iterator {
        void operator++() { ++it; }
        bool operator==(const Iterator& other) const { return it == other.it; }
        xla::ClientAndPtr<xla::PjRtDevice> operator*() const {
          return xla::WrapWithClient(py_client, *it);
        }
        const std::shared_ptr<xla::PyClient>& py_client;
        xla::ifrt::DeviceList::Devices::const_iterator it;
      };
      return py::make_iterator(
          Iterator{py_client_, std::get<0>(device_list_).begin()},
          Iterator{py_client_, std::get<0>(device_list_).end()});
    }
    case 1:
      return py::make_iterator(std::get<1>(device_list_).begin(),
                               std::get<1>(device_list_).end());
    default:
      throw py::value_error("Unrecognized DeviceList type");
  }
}

std::string PyDeviceList::Str() { return py::str(AsTuple()); }

py::tuple PyDeviceList::Dump() { return AsTuple(); }

std::shared_ptr<PyDeviceList> PyDeviceList::Load(
    py::tuple py_device_assignment) {
  return std::make_shared<PyDeviceList>(std::move(py_device_assignment));
}

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
        for (py::handle device : std::get<1>(device_list_)) {
          if (py::cast<int>(device.attr("process_index")) !=
              py::cast<int>(device.attr("client").attr("process_index")())) {
            is_fully_addressable_ = false;
            break;
          }
        }
        break;
      }
      default:
        throw py::value_error("Unrecognized DeviceList type");
    }
  }
  return *is_fully_addressable_;
}

std::shared_ptr<PyDeviceList> PyDeviceList::AddressableDeviceList() {
  if (IsFullyAddressable()) {
    // Do not cache this result in `addressable_device_list_`. Otherwise, it
    // will create a cycle that prevents deletion of this object.
    return shared_from_this();
  }
  if (!addressable_device_list_.has_value()) {
    switch (device_list_.index()) {
      case 0: {
        xla::ifrt::DeviceList::Devices addressable_devices;
        const int process_index = py_client_ ? py_client_->process_index() : 0;
        for (xla::ifrt::Device* device : std::get<0>(device_list_).devices()) {
          if (device->process_index() == process_index) {
            addressable_devices.push_back(device);
          }
        }
        addressable_device_list_ = std::make_shared<PyDeviceList>(
            py_client_, xla::ifrt::DeviceList(std::move(addressable_devices)));
        break;
      }
      case 1: {
        std::vector<py::object> addressable_py_device_assignment;
        for (py::handle device : std::get<1>(device_list_)) {
          if (py::cast<int>(device.attr("process_index")) ==
              py::cast<int>(device.attr("client").attr("process_index")())) {
            addressable_py_device_assignment.push_back(
                py::cast<py::object>(device));
          }
        }
        addressable_device_list_ = std::make_shared<PyDeviceList>(
            py::cast(std::move(addressable_py_device_assignment)));
        break;
      }
      default:
        throw py::value_error("Unrecognized DeviceList type");
    }
  }
  return *addressable_device_list_;
}

void PyDeviceList::PopulateMemoryKindInfo() {
  if (device_list_.index() == 1) {
    // Handle Python duck-type devices in a separate function for readability.
    PopulateMemoryKindInfoForDuckTypedDevices();
    return;
  }
  if (device_list_.index() != 0) {
    throw py::value_error("Unrecognized DeviceList type");
  }
  MemoryKindInfo info;
  if (!GetEnableMemories()) {
    info.default_memory_kind = py::none();
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
    info.default_memory_kind = py::none();
    memory_kind_info_ = std::move(info);
    return;
  }

  auto default_memory = addressable_device->default_memory_space();
  if (!default_memory.ok()) {
    // Cache the error.
    memory_kind_info_ = default_memory.status();
    return;
  }
  info.default_memory_kind =
      py::cast(std::string((*default_memory)->memory_space_kind()));
  std::vector<std::string> memory_kinds;
  memory_kinds.reserve(addressable_device->memory_spaces().size());
  for (xla::ifrt::Memory* memory : addressable_device->memory_spaces()) {
    memory_kinds.push_back(std::string(memory->memory_space_kind()));
  }
  info.memory_kinds = py::cast(memory_kinds);
  memory_kind_info_ = std::move(info);
}

void PyDeviceList::PopulateMemoryKindInfoForDuckTypedDevices() {
  MemoryKindInfo info;
  if (!GetEnableMemories()) {
    info.default_memory_kind = py::none();
    // info.memory_kinds is default-initialized to an empty tuple.
    memory_kind_info_ = std::move(info);
    return;
  }
  try {
    py::handle addressable_device;
    for (py::handle device : std::get<1>(device_list_)) {
      if (py::cast<int>(device.attr("process_index")) ==
          py::cast<int>(device.attr("client").attr("process_index")())) {
        addressable_device = device;
        break;
      }
    }
    if (!addressable_device) {
      info.default_memory_kind = py::none();
      // info.memory_kinds is default-initialized to an empty tuple.
      memory_kind_info_ = std::move(info);
      return;
    }
    auto default_memory = addressable_device.attr("default_memory")();
    info.default_memory_kind = default_memory.attr("kind");
    info.memory_kinds = addressable_device.attr("addressable_memories")();
    memory_kind_info_ = std::move(info);
  } catch (py::error_already_set& e) {
    // Cache the error.
    memory_kind_info_ = xla::InvalidArgument("%s", e.what());
  }
}

xla::StatusOr<py::tuple> PyDeviceList::MemoryKinds() {
  if (!memory_kind_info_.has_value()) {
    PopulateMemoryKindInfo();
  }
  if (!memory_kind_info_->ok()) {
    return memory_kind_info_->status();
  }
  return (*memory_kind_info_)->memory_kinds;
}

xla::StatusOr<py::object> PyDeviceList::DefaultMemoryKind() {
  if (!memory_kind_info_.has_value()) {
    PopulateMemoryKindInfo();
  }
  if (!memory_kind_info_->ok()) {
    return memory_kind_info_->status();
  }
  return (*memory_kind_info_)->default_memory_kind;
}

void RegisterDeviceList(py::module& m) {
  py::class_<PyDeviceList, std::shared_ptr<PyDeviceList>>(m, "DeviceList")
      .def(py::init<py::tuple>())
      .def("__hash__", &PyDeviceList::Hash)
      .def("__eq__", &PyDeviceList::operator==)
      .def("__ne__", &PyDeviceList::operator!=)
      .def("__len__", &PyDeviceList::Len)
      .def("__getitem__", &PyDeviceList::GetItem)
      .def("__getitem__", &PyDeviceList::GetSlice)
      .def("__iter__", &PyDeviceList::Iter, py::keep_alive<0, 1>())
      .def("__str__", &PyDeviceList::Str)
      .def(py::pickle([](PyDeviceList* l) { return l->Dump(); },
                      [](py::tuple t) { return PyDeviceList::Load(t); }))
      .def_property_readonly("is_fully_addressable",
                             &PyDeviceList::IsFullyAddressable)
      .def_property_readonly("addressable_device_list",
                             &PyDeviceList::AddressableDeviceList)
      // `xla::ValueOrThrowWrapper` does not work with
      // `def_property_readonly()`. Manually convert an error into an exception.
      .def_property_readonly(
          "default_memory_kind",
          [](PyDeviceList* l) {
            auto kind = l->DefaultMemoryKind();
            if (!kind.ok()) {
              throw py::value_error(kind.status().ToString());
            }
            return *kind;
          })
      .def_property_readonly("memory_kinds", [](PyDeviceList* l) {
        auto kinds = l->MemoryKinds();
        if (!kinds.ok()) {
          throw py::value_error(kinds.status().ToString());
        }
        return *kinds;
      });
}

}  // namespace jax
