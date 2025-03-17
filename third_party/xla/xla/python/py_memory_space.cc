/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/py_memory_space.h"

#include <Python.h>

#include <utility>

#include "absl/strings/string_view.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/py_client.h"

namespace nb = ::nanobind;

namespace xla {

PyMemorySpace::PyMemorySpace(nb_class_ptr<PyClient> client,
                             ifrt::Memory* memory)
    : client_(std::move(client)), memory_(memory) {}

int PyMemorySpace::process_index() const { return client_->process_index(); }

absl::string_view PyMemorySpace::platform() const {
  // TODO(phawkins): this is a temporary backwards
  // compatibility shim. We changed the name PJRT
  // reports for GPU platforms to "cuda" or "rocm",
  // but we haven't yet updated JAX clients that
  // expect "gpu". Migrate users and remove this
  // code.
  if (client_->platform_name() == "cuda" ||
      client_->platform_name() == "rocm") {
    return absl::string_view("gpu");
  } else {
    return client_->platform_name();
  }
}

absl::string_view PyMemorySpace::kind() const {
  return *memory_->Kind().memory_kind();
}

absl::string_view PyMemorySpace::Str() const { return memory_->DebugString(); }

absl::string_view PyMemorySpace::Repr() const { return memory_->ToString(); }

nb::list PyMemorySpace::AddressableByDevices() const {
  nb::list devices;
  for (ifrt::Device* device : memory_->Devices()) {
    devices.append(client_->GetPyDevice(device));
  }
  return devices;
}

/* static */ int PyMemorySpace::tp_traverse(PyObject* self, visitproc visit,
                                            void* arg) {
  PyMemorySpace* d = nb::inst_ptr<PyMemorySpace>(self);
  Py_VISIT(d->client().ptr());
  return 0;
}

/* static */ int PyMemorySpace::tp_clear(PyObject* self) {
  PyMemorySpace* d = nb::inst_ptr<PyMemorySpace>(self);
  nb_class_ptr<PyClient> client;
  std::swap(client, d->client_);
  return 0;
}

PyType_Slot PyMemorySpace::slots_[] = {
    {Py_tp_traverse, (void*)PyMemorySpace::tp_traverse},
    {Py_tp_clear, (void*)PyMemorySpace::tp_clear},
    {0, nullptr},
};

/* static */ void PyMemorySpace::RegisterPythonType(nb::module_& m) {
  nb::class_<PyMemorySpace> device(m, "Memory",
                                   nb::type_slots(PyMemorySpace::slots_));
  device.def_prop_ro("process_index", &PyMemorySpace::process_index)
      .def_prop_ro("platform", &PyMemorySpace::platform)
      .def_prop_ro("kind", &PyMemorySpace::kind)
      .def("__str__", &PyMemorySpace::Str)
      .def("__repr__", &PyMemorySpace::Repr)
      .def("addressable_by_devices", &PyMemorySpace::AddressableByDevices,
           "Returns devices that can address this memory.");
}

}  // namespace xla
