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

#ifndef XLA_PYTHON_PY_DEVICE_LIST_H_
#define XLA_PYTHON_PY_DEVICE_LIST_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/py_client.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace jax {

// Device list with various caching and direct access to IFRT DeviceList.
class PyDeviceList {
 public:
  PyDeviceList(xla::nb_class_ptr<xla::PyClient> py_client,
               tsl::RCReference<xla::ifrt::DeviceList> device_list);
  explicit PyDeviceList(nanobind::tuple py_device_assignment);
  ~PyDeviceList();

  PyDeviceList(const PyDeviceList&) = delete;
  PyDeviceList(PyDeviceList&&) = delete;
  PyDeviceList& operator=(const PyDeviceList&) = delete;
  PyDeviceList& operator=(PyDeviceList&&) = delete;

  static nanobind::handle type() {
    static auto type = nanobind::type<PyDeviceList>();
    return type;
  }

  // These two methods are safe to call from C++ without GIL.
  xla::nb_class_ptr<xla::PyClient> py_client() const { return py_client_; }
  absl::StatusOr<tsl::RCReference<xla::ifrt::DeviceList>> ifrt_device_list()
      const;

  int Len() const;                      // Requires the GIL in GIL mode.
  nanobind::object GetItem(int index);  // Requires the GIL in GIL mode.

  // Requires the GIL in GIL mode. Acquires the self lock in non-GIL mode.
  static xla::nb_class_ptr<PyDeviceList> AddressableDeviceList(
      xla::nb_class_ptr<PyDeviceList> self);

  // Requires the GIL in GIL mode. Acquires the self lock in non-GIL mode.
  static absl::StatusOr<nanobind::object> DefaultMemoryKind(
      xla::nb_class_ptr<PyDeviceList> self);

  // Requires the GIL in GIL mode. Acquires the self lock in non-GIL mode.
  static absl::StatusOr<nanobind::tuple> MemoryKinds(
      xla::nb_class_ptr<PyDeviceList> self);

  // go/pywald-pybind-annotation BEGIN
  // refs {
  //   module_path: "third_party/tensorflow/compiler/xla/python/xla.cc"
  //   module_arg {}
  // }
  // go/pywald-pybind-annotation END
  static void Register(nanobind::module_& m);

 private:
  nanobind::tuple AsTuple() const;

  // Methods below require GIL.
  nanobind::object GetSlice(nanobind::slice slice);
  nanobind::iterator Iter();

  std::string Str();

  nanobind::tuple Dump() const;

  int64_t Hash();  // Mutates hash_, needs self lock.

  static bool Equal(xla::nb_class_ptr<PyDeviceList> self,
                    nanobind::handle other);
  static bool NotEqual(xla::nb_class_ptr<PyDeviceList> self,
                       nanobind::handle other);

  // Finds the memory kind info from an addressable device. Requires the GIL
  // or self lock.
  void PopulateMemoryKindInfo();
  // Same as `PopulateMemoryKindInfo()`, but uses `py_device_assignment_`
  // instead of `ifrt_device_list_` to support duck-typed device objects.
  // Requires the GIL or self lock.
  void PopulateMemoryKindInfoForDuckTypedDevices();

  // Requires the self lock or GIL is held.
  bool IsFullyAddressable();

  // Valid only if `device_list_` contains `xla::ifrt::DeviceList` and
  // non-empty.
  xla::nb_class_ptr<xla::PyClient> py_client_;

  // Either C++ `ifrt::DeviceList` or Python duck-type devices.
  // TODO(hyeontaek): Remove support for Python duck-type devices once all
  // JAX backends and tests are migrated to use an `xla::ifrt::Device` type
  // for JAX devices.
  // Immutable after constructor; no locking needed.
  std::variant<tsl::RCReference<xla::ifrt::DeviceList>, nanobind::tuple>
      device_list_;

  // Populated on demand. Guarded by the object's self lock.
  std::optional<ssize_t> hash_;
  // TODO(hyeontaek): Make the following property cached within
  // `xla::ifrt::DeviceList`.
  // Populated on demand. Guarded by the object's self lock.
  std::optional<bool> is_fully_addressable_;
  // Populated on demand. Guarded by the object's self lock.
  std::optional<xla::nb_class_ptr<PyDeviceList>> addressable_device_list_;

  struct MemoryKindInfo {
    nanobind::object default_memory_kind;
    nanobind::tuple memory_kinds;
  };
  // Populated on demand. Guarded by the object's self lock.
  std::optional<absl::StatusOr<MemoryKindInfo>> memory_kind_info_;
};

}  // namespace jax

#endif  // XLA_PYTHON_PY_DEVICE_LIST_H_
