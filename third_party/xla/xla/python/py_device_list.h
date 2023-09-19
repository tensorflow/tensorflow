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

#ifndef XLA_PYTHON_PY_DEVICE_LIST_H_
#define XLA_PYTHON_PY_DEVICE_LIST_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "xla/python/ifrt/device.h"
#include "xla/python/py_client.h"
#include "xla/statusor.h"

namespace jax {

// Device list with various caching and direct access to IFRT DeviceList.
class PyDeviceList : public std::enable_shared_from_this<PyDeviceList> {
 public:
  PyDeviceList(std::shared_ptr<xla::PyClient> py_client,
               xla::ifrt::DeviceList device_list);
  explicit PyDeviceList(pybind11::tuple py_device_assignment);
  ~PyDeviceList();

  PyDeviceList(const PyDeviceList&) = delete;
  PyDeviceList(PyDeviceList&&) = delete;
  PyDeviceList& operator=(const PyDeviceList&) = delete;
  PyDeviceList& operator=(PyDeviceList&&) = delete;

  // These two methods are safe to call from C++ without GIL.
  std::shared_ptr<xla::PyClient> py_client() const { return py_client_; }
  xla::StatusOr<xla::ifrt::DeviceList> ifrt_device_list() const;

  // Methods below require GIL.
  int64_t Hash();
  bool operator==(pybind11::handle other);
  bool operator!=(pybind11::handle other);

  int Len() const;
  pybind11::object GetItem(int index);
  pybind11::object GetSlice(pybind11::slice slice);
  pybind11::iterator Iter();

  std::string Str();

  pybind11::tuple Dump();
  static std::shared_ptr<PyDeviceList> Load(
      pybind11::tuple py_device_assignment);

  bool IsFullyAddressable();
  std::shared_ptr<PyDeviceList> AddressableDeviceList();
  xla::StatusOr<pybind11::object> DefaultMemoryKind();
  xla::StatusOr<pybind11::tuple> MemoryKinds();

 private:
  pybind11::tuple AsTuple();

  // Finds the memory kind info from an addressable device.
  void PopulateMemoryKindInfo();
  // Same as `PopulateMemoryKindInfo()`, but uses `py_device_assignment_`
  // instead of `ifrt_device_list_` to support duck-typed device objects.
  void PopulateMemoryKindInfoForDuckTypedDevices();

  // Valid only if `device_list_` contains `xla::ifrt::DeviceList` and
  // non-empty.
  std::shared_ptr<xla::PyClient> py_client_;

  // Either C++ `ifrt::DeviceList` or Python duck-type devices.
  // TODO(hyeontaek): Remove support for Python duck-type devices once all
  // JAX backends and tests are migrated to use an `xla::ifrt::Device` type
  // for JAX devices.
  std::variant<xla::ifrt::DeviceList, pybind11::tuple> device_list_;

  std::optional<ssize_t> hash_;  // Populated on demand.
  // TODO(hyeontaek): Make the following property cached within
  // `xla::ifrt::DeviceList`.
  std::optional<bool> is_fully_addressable_;  // Populated on demand.
  std::optional<std::shared_ptr<PyDeviceList>>
      addressable_device_list_;  // Populated on demand.

  struct MemoryKindInfo {
    pybind11::object default_memory_kind;
    pybind11::tuple memory_kinds;
  };
  std::optional<xla::StatusOr<MemoryKindInfo>>
      memory_kind_info_;  // Populated on demand.
};

// go/pywald-pybind-annotation BEGIN
// refs {
//   module_path: "third_party/tensorflow/compiler/xla/python/xla.cc"
//   module_arg {}
// }
// go/pywald-pybind-annotation END
void RegisterDeviceList(pybind11::module& m);

}  // namespace jax

#endif  // XLA_PYTHON_PY_DEVICE_LIST_H_
