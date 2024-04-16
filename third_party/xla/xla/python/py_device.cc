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

#include "xla/python/py_device.h"

#include <Python.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/variant.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_helpers.h"
#include "xla/python/py_client.h"
#include "xla/python/py_memory_space.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/framework/allocator.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace nb = ::nanobind;

namespace xla {

PyDevice::PyDevice(nb_class_ptr<PyClient> client, ifrt::Device* device)
    : client_(std::move(client)), device_(device) {}

int PyDevice::id() const { return device_->id(); }

int PyDevice::process_index() const { return device_->process_index(); }

std::string_view PyDevice::platform() const {
  // TODO(phawkins): this is a temporary backwards
  // compatibility shim. We changed the name PJRT
  // reports for GPU platforms to "cuda" or "rocm",
  // but we haven't yet updated JAX clients that
  // expect "gpu". Migrate users and remove this
  // code.
  if (client_->platform_name() == "cuda" ||
      client_->platform_name() == "rocm") {
    return std::string_view("gpu");
  } else {
    return client_->platform_name();
  }
}

std::string_view PyDevice::device_kind() const {
  return device_->device_kind();
}

std::optional<int> PyDevice::local_hardware_id() const {
  int local_hardware_id = device_->local_hardware_id();
  if (local_hardware_id == -1) {
    return std::nullopt;
  }
  return local_hardware_id;
}

std::string_view PyDevice::Str() const { return device_->DebugString(); }

std::string_view PyDevice::Repr() const { return device_->ToString(); }

absl::Status PyDevice::TransferToInfeed(LiteralSlice literal) {
  GlobalPyRefManager()->CollectGarbage();
  nb::gil_scoped_release gil_release;
  return device_->TransferToInfeed(literal);
}

absl::StatusOr<nb::object> PyDevice::TransferFromOutfeed(Shape shape) {
  GlobalPyRefManager()->CollectGarbage();
  std::shared_ptr<Literal> literal;
  {
    nb::gil_scoped_release gil_release;
    ShapeUtil::ForEachMutableSubshape(
        &shape, [](Shape* subshape, const ShapeIndex&) {
          if (!subshape->has_layout()) {
            LayoutUtil::SetToDefaultLayout(subshape);
          }
        });
    literal = std::make_shared<Literal>(shape);
    TF_RETURN_IF_ERROR(device_->TransferFromOutfeed(literal.get()));
  }
  return LiteralToPython(std::move(literal));
}

absl::StatusOr<nb_class_ptr<PyMemorySpace>> PyDevice::Memory(
    std::string_view kind) const {
  xla::PjRtMemorySpace* result_memory_space = nullptr;
  for (auto* memory_space : device_->memory_spaces()) {
    if (memory_space->kind() == kind) {
      if (result_memory_space != nullptr) {
        std::string memories =
            absl::StrJoin(device_->memory_spaces(), ", ",
                          [](std::string* out, const auto& memory_space) {
                            absl::StrAppend(out, memory_space->kind());
                          });
        auto device_kind = device_->device_kind();
        return xla::InvalidArgument(
            "Found more than one addressable memory for "
            "kind %s which is not allowed. There can only "
            "be one memory for each "
            "kind. Device %s can address the following "
            "memory kinds: %s",
            kind, device_kind, memories);
      }
      result_memory_space = memory_space;
    }
  }
  if (result_memory_space == nullptr) {
    std::string memories =
        absl::StrJoin(device_->memory_spaces(), ", ",
                      [](std::string* out, const auto& memory_space) {
                        absl::StrAppend(out, memory_space->kind());
                      });
    auto device_kind = device_->device_kind();
    return xla::InvalidArgument(
        "Could not find memory addressable by device %s. Device %s "
        "can address the following memory kinds: %s. "
        "Got memory kind: %s",
        device_kind, device_kind, memories, kind);
  }
  return client_->GetPyMemorySpace(result_memory_space);
}

absl::StatusOr<nb_class_ptr<PyMemorySpace>> PyDevice::DefaultMemory() const {
  TF_ASSIGN_OR_RETURN(auto* memory_space, device_->default_memory_space());
  return client_->GetPyMemorySpace(memory_space);
}

nb::list PyDevice::AddressableMemories() const {
  nb::list memory_spaces;
  for (auto* memory_space : device_->memory_spaces()) {
    memory_spaces.append(client_->GetPyMemorySpace(memory_space));
  }
  return memory_spaces;
}

absl::StatusOr<std::optional<nb::dict>> PyDevice::MemoryStats() const {
  GlobalPyRefManager()->CollectGarbage();
  absl::StatusOr<tsl::AllocatorStats> maybe_stats =
      device_->GetAllocatorStats();
  if (absl::IsUnimplemented(maybe_stats.status())) {
    return std::nullopt;
  }
  // Raise error if any status other than Unimplemented is returned.
  ThrowIfError(maybe_stats.status());

  nb::dict result;
  result["num_allocs"] = maybe_stats->num_allocs;
  result["bytes_in_use"] = maybe_stats->bytes_in_use;
  result["peak_bytes_in_use"] = maybe_stats->peak_bytes_in_use;
  result["largest_alloc_size"] = maybe_stats->largest_alloc_size;
  if (maybe_stats->bytes_limit) {
    result["bytes_limit"] = *maybe_stats->bytes_limit;
  }
  result["bytes_reserved"] = maybe_stats->bytes_reserved;
  result["peak_bytes_reserved"] = maybe_stats->peak_bytes_reserved;
  if (maybe_stats->bytes_reservable_limit) {
    result["bytes_reservable_limit"] = *maybe_stats->bytes_reservable_limit;
  }
  result["largest_free_block_bytes"] = maybe_stats->largest_free_block_bytes;
  if (maybe_stats->pool_bytes) {
    result["pool_bytes"] = *maybe_stats->pool_bytes;
  }
  if (maybe_stats->peak_pool_bytes) {
    result["peak_pool_bytes"] = *maybe_stats->peak_pool_bytes;
  }
  return result;
}

absl::StatusOr<std::intptr_t> PyDevice::GetStreamForExternalReadyEvents()
    const {
  return device_->GetStreamForExternalReadyEvents();
}

/* static */ int PyDevice::tp_traverse(PyObject* self, visitproc visit,
                                       void* arg) {
  PyDevice* d = nb::inst_ptr<PyDevice>(self);
  Py_VISIT(d->client().ptr());
  return 0;
}

/* static */ int PyDevice::tp_clear(PyObject* self) {
  PyDevice* d = nb::inst_ptr<PyDevice>(self);
  nb_class_ptr<PyClient> client;
  std::swap(client, d->client_);
  return 0;
}

PyType_Slot PyDevice::slots_[] = {
    {Py_tp_traverse, (void*)PyDevice::tp_traverse},
    {Py_tp_clear, (void*)PyDevice::tp_clear},
    {0, nullptr},
};

/* static */ void PyDevice::RegisterPythonType(nb::module_& m) {
  nb::class_<PyDevice> device(
      m, "Device", nb::type_slots(PyDevice::slots_),
      "A descriptor of an available device.\n\nSubclasses are used to "
      "represent specific types of devices, e.g. CPUs, GPUs. Subclasses may "
      "have additional properties specific to that device type.");
  device
      .def_prop_ro(
          "id", &PyDevice::id,
          "Integer ID of this device.\n\nUnique across all available devices "
          "of this type, including remote devices on multi-host platforms.")
      .def_prop_ro("process_index", &PyDevice::process_index,
                   "Integer index of this device's process.\n\n"
                   "This is always 0 except on multi-process platforms.")
      .def_prop_ro("host_id", &PyDevice::process_index,
                   "Deprecated; please use process_index")
      .def_prop_ro("task_id", &PyDevice::process_index,
                   "Deprecated; please use process_index")
      .def_prop_ro("platform", &PyDevice::platform)
      .def_prop_ro("device_kind", &PyDevice::device_kind)
      .def_prop_ro("client", &PyDevice::client)
      .def_prop_ro(
          "local_hardware_id", &PyDevice::local_hardware_id,
          "Opaque hardware ID, e.g., the CUDA device number. In general, not "
          "guaranteed to be dense, and not guaranteed to be defined on all "
          "platforms.")
      .def("__str__", &PyDevice::Str)
      .def("__repr__", &PyDevice::Repr)
      .def("transfer_to_infeed",
           ThrowIfErrorWrapper(&PyDevice::TransferToInfeed))
      .def("transfer_from_outfeed",
           ValueOrThrowWrapper(&PyDevice::TransferFromOutfeed))
      .def("memory", ValueOrThrowWrapper(&PyDevice::Memory), nb::arg("kind"))
      .def("default_memory", ValueOrThrowWrapper(&PyDevice::DefaultMemory),
           "Returns the default memory of a device.")
      .def("addressable_memories", &PyDevice::AddressableMemories,
           "Returns all the memories that a device can address.")

      .def("live_buffers",
           [](nb::handle device) {
             PythonDeprecationWarning(
                 "Per device live_buffers() is deprecated. Please "
                 "use the jax.live_arrays() for jax.Arrays instead.");
             return nb::list();
           })
      .def(
          "memory_stats", ValueOrThrowWrapper(&PyDevice::MemoryStats),
          "Returns memory statistics for this device keyed by name. May not "
          "be implemented on all platforms, and different platforms may return "
          "different stats, or -1 for unavailable stats. 'bytes_in_use' is "
          "usually available. Intended for diagnostic use.")
      .def(
          "get_stream_for_external_ready_events",
          xla::ValueOrThrowWrapper(&PyDevice::GetStreamForExternalReadyEvents));
  static PyMethodDef get_attr_method = {
      "__getattr__",
      +[](PyObject* self, PyObject* args) -> PyObject* {
        PyObject* key;
        if (!PyArg_ParseTuple(args, "O", &key)) {
          PyErr_SetString(PyExc_TypeError, "__getattr__ must take 1 argument.");
          return nullptr;
        }
        try {
          auto device = nb::cast<PyDevice*>(nb::handle(self));
          auto name = nb::cast<std::string_view>(nb::handle(key));
          const auto& attrs = device->device_->Attributes();
          auto it = attrs.find(name);
          if (it != attrs.end()) {
            auto result =
                std::visit([](auto&& v) { return nb::cast(v); }, it->second);
            return result.release().ptr();
          }
          PyErr_SetNone(PyExc_AttributeError);
          return nullptr;
        } catch (std::exception& e) {
          PyErr_Format(PyExc_SystemError, "Unhandled nanobind exception: %s",
                       e.what());
          return nullptr;
        } catch (...) {
          PyErr_SetString(PyExc_SystemError, "Unhandled nanobind exception.");
          return nullptr;
        }
      },
      METH_VARARGS,
      nullptr,
  };
  device.attr("__getattr__") = nb::steal<nb::object>(PyDescr_NewMethod(
      reinterpret_cast<PyTypeObject*>(device.ptr()), &get_attr_method));
}

}  // namespace xla
