/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/python/xla.h"

#include <Python.h>

#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// clang-format off
#include "absl/base/casts.h"
// Must be included first
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/nb_defs.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/pair.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/variant.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "xla/ffi/ffi_api.h"
#include "xla/layout_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/py_client.h"
#include "xla/service/cpu/collectives_interface.h"
#include "tsl/python/lib/core/numpy.h"  //NOLINT
#ifdef XLA_PYTHON_ENABLE_GPU
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#endif  // XLA_PYTHON_ENABLE_GPU

#ifdef __linux__
#include "third_party/gloo/gloo/transport/tcp/attr.h"
#include "third_party/gloo/gloo/transport/tcp/device.h"
#include "xla/pjrt/cpu/gloo_collectives.h"
#include "xla/pjrt/cpu/gloo_kv_store.h"
#endif  // __linux__

#include "xla/literal.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/custom_call_sharding.h"
#include "xla/python/dlpack.h"
#include "xla/python/jax_jit.h"
#include "xla/python/logging.h"
#include "xla/python/mlir.h"
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/ops.h"
#include "xla/python/outfeed_receiver_py.h"
#include "xla/python/pjit.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pmap_lib.h"
#include "xla/python/pprof_profile_builder.h"
#include "xla/python/profiler.h"
#include "xla/python/py_array.h"
#include "xla/python/py_compile_only_client.h"
#include "xla/python/py_device_list.h"
#include "xla/python/py_executable.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/pytree.h"
#include "xla/python/sharding.h"
#include "xla/python/traceback.h"
#include "xla/python/transfer_guard_lib.h"
#include "xla/python/types.h"
#include "xla/python/util.h"
#include "xla/python/weakref_lru_cache.h"
#include "xla/python/xla_compiler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "tsl/distributed_runtime/preemption/preemption_sync_manager.h"
#include "tsl/platform/platform.h"

// TODO(phawkins): remove host_id properties after JAX is update to avoid them.

namespace xla {
namespace {

namespace nb = nanobind;
namespace py = pybind11;

bool IsOptimizedBuild() {
#if NDEBUG
  return true;
#else
  return false;
#endif  // NDEBUG
}

// Is*san reports whether the build is under that particular sanitizer.
bool IsAsan() {
#if defined(ADDRESS_SANITIZER)
  return true;
#else  // defined(ADDRESS_SANITIZER)
  return false;
#endif
}

bool IsMsan() {
#if defined(MEMORY_SANITIZER)
  return true;
#else  // defined(MEMORY_SANITIZER)
  return false;
#endif
}

bool IsTsan() {
#if defined(THREAD_SANITIZER)
  return true;
#else  // defined(THREAD_SANITIZER)
  return false;
#endif
}

// IsSanitized reports whether the build is under any sanitizer.
bool IsSanitized() { return IsAsan() || IsMsan() || IsTsan(); }

}  // namespace

static void Init(py::module_& m) {
  // Initialize ABSL logging because code within XLA uses it.
#ifndef PLATFORM_GOOGLE
  InitializeAbslLogging();
#endif  // PLATFORM_GOOGLE

  // Normally this would happen at the start of NB_MODULE, but since this is a
  // pybind11 module we have to do this ourselves.
  nb::detail::init(NB_DOMAIN_STR);

  // We seem to get a fair number of leak warnings from nanobind. It's unclear
  // whether these are false positives or not.
  nb::set_leak_warnings(false);

  tsl::ImportNumpy();

  nb::module_ m_nb = nb::cast<nb::module_>(nb::borrow(m.ptr()));

  // Exceptions
  py::register_exception<XlaRuntimeError>(m, "XlaRuntimeError",
                                          PyExc_RuntimeError);

  // TODO(phawkins): use nb::exception<> once we have migrated all the pybind11
  // code to nanobind. We use nb::register_exception_translator because we don't
  // want to define the exception twice.
  nb::register_exception_translator(
      [](const std::exception_ptr& p, void* payload) {
        try {
          std::rethrow_exception(p);
        } catch (const XlaRuntimeError& e) {
          PyErr_SetString(reinterpret_cast<PyObject*>(payload), e.what());
        }
      },
      nb::getattr(m_nb, "XlaRuntimeError").ptr());

  // Types
  nb::enum_<PrimitiveType>(m_nb, "PrimitiveType")
      .value("PRIMITIVE_TYPE_INVALID", PRIMITIVE_TYPE_INVALID)
      .value("PRED", PRED)
      .value("S4", S4)
      .value("S8", S8)
      .value("S16", S16)
      .value("S32", S32)
      .value("S64", S64)
      .value("U4", U4)
      .value("U8", U8)
      .value("U16", U16)
      .value("U32", U32)
      .value("U64", U64)
      .value("F16", F16)
      .value("F8E4M3FN", F8E4M3FN)
      .value("F8E4M3B11FNUZ", F8E4M3B11FNUZ)
      .value("F8E4M3FNUZ", F8E4M3FNUZ)
      .value("F8E5M2", F8E5M2)
      .value("F8E5M2FNUZ", F8E5M2FNUZ)
      .value("BF16", BF16)
      .value("F32", F32)
      .value("F64", F64)
      .value("C64", C64)
      .value("C128", C128)
      .value("TUPLE", TUPLE)
      .value("OPAQUE_TYPE", OPAQUE_TYPE)
      .value("TOKEN", TOKEN);

  // Must be before PyClient.compile.
  BuildXlaCompilerSubmodule(m_nb);

  py::class_<PjRtDevice, ClientAndPtr<PjRtDevice>> device(
      m, "Device",
      "A descriptor of an available device.\n\nSubclasses are used to "
      "represent specific types of devices, e.g. CPUs, GPUs. Subclasses may "
      "have additional properties specific to that device type.");
  device
      .def_property_readonly(
          "id", &PjRtDevice::id,
          "Integer ID of this device.\n\nUnique across all available devices "
          "of this type, including remote devices on multi-host platforms.")
      .def_property_readonly(
          "process_index", &PjRtDevice::process_index,
          "Integer index of this device's process.\n\n"
          "This is always 0 except on multi-process platforms.")
      .def_property_readonly("host_id", &PjRtDevice::process_index,
                             "Deprecated; please use process_index")
      .def_property_readonly("task_id", &PjRtDevice::process_index,
                             "Deprecated; please use process_index")
      .def_property_readonly("platform",
                             [](const ClientAndPtr<PjRtDevice>& device) {
                               // TODO(phawkins): this is a temporary backwards
                               // compatibility shim. We changed the name PJRT
                               // reports for GPU platforms to "cuda" or "rocm",
                               // but we haven't yet updated JAX clients that
                               // expect "gpu". Migrate users and remove this
                               // code.
                               if (device.client()->platform_name() == "cuda" ||
                                   device.client()->platform_name() == "rocm") {
                                 return absl::string_view("gpu");
                               } else {
                                 return device.client()->platform_name();
                               }
                             })
      .def_property_readonly("device_kind", &PjRtDevice::device_kind)
      .def_property_readonly("client",
                             [](const ClientAndPtr<PjRtDevice>& device) {
                               return device.client();
                             })
      .def_property_readonly(
          "local_hardware_id",
          [](const ClientAndPtr<PjRtDevice>& device) -> std::optional<int> {
            int local_hardware_id = device->local_hardware_id();
            if (local_hardware_id == -1) {
              return std::nullopt;
            }
            return local_hardware_id;
          },
          "Opaque hardware ID, e.g., the CUDA device number. In general, not "
          "guaranteed to be dense, and not guaranteed to be defined on all "
          "platforms.")
      .def("__str__", &PjRtDevice::DebugString)
      .def("__repr__", &PjRtDevice::ToString)
      .def("transfer_to_infeed",
           [](PjRtDevice& device, py::handle literal_py) {
             // TODO(phawkins): just accept a Shape argument after nanobind
             // transition is complete.
             // We use a type caster directly because we need the value to
             // alive until the transfer completes.
             nb::detail::type_caster<LiteralSlice> literal_caster;
             if (!literal_caster.from_python(literal_py.ptr(), 0, nullptr)) {
               throw py::cast_error();
             }
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             xla::ThrowIfError(device.TransferToInfeed(literal_caster.value));
           })
      .def("transfer_from_outfeed",
           [](PjRtDevice& device, py::handle shape_py) -> py::object {
             // TODO(phawkins): just accept a Shape argument after nanobind
             // transition is complete.
             Shape shape = nb::cast<Shape>(nb::borrow(shape_py.ptr()));
             GlobalPyRefManager()->CollectGarbage();
             std::shared_ptr<Literal> literal;
             {
               py::gil_scoped_release gil_release;
               ShapeUtil::ForEachMutableSubshape(
                   &shape, [](Shape* subshape, const ShapeIndex&) {
                     if (!subshape->has_layout()) {
                       LayoutUtil::SetToDefaultLayout(subshape);
                     }
                   });
               literal = std::make_shared<Literal>(shape);
               xla::ThrowIfError(device.TransferFromOutfeed(literal.get()));
             }
             nb::object out = ValueOrThrow(LiteralToPython(std::move(literal)));
             return py::reinterpret_steal<py::object>(out.release().ptr());
           })
      .def(
          "memory",
          [](const ClientAndPtr<PjRtDevice>& device, const std::string& kind) {
            return jax::GetMemory(device, kind);
          },
          py::arg("kind"))
      // Returns the default memory of a device.
      .def("default_memory",
           [](const ClientAndPtr<PjRtDevice>& device) {
             auto* memory_space =
                 xla::ValueOrThrow(device->default_memory_space());
             return WrapWithClient(device.client(), memory_space);
           })
      // Returns all the memories that a device can address.
      .def("addressable_memories",
           [](const ClientAndPtr<PjRtDevice>& device) {
             std::vector<ClientAndPtr<PjRtMemorySpace>> memory_spaces;
             auto span = device->memory_spaces();
             memory_spaces.reserve(span.size());
             for (auto* memory_space : span) {
               memory_spaces.push_back(
                   WrapWithClient(device.client(), memory_space));
             }
             return memory_spaces;
           })
      .def("live_buffers",
           [](const ClientAndPtr<PjRtDevice>& device) {
             PythonDeprecationWarning(
                 "Per device live_buffers() is going to be deprecated. Please "
                 "use the jax.live_arrays() for jax.Arrays instead.");
             return py::list();
           })
      .def(
          "memory_stats",
          [](const PjRtDevice& device)
              -> std::optional<std::map<std::string, int64_t>> {
            GlobalPyRefManager()->CollectGarbage();
            xla::StatusOr<tsl::AllocatorStats> maybe_stats =
                device.GetAllocatorStats();
            if (absl::IsUnimplemented(maybe_stats.status())) {
              return std::nullopt;
            }
            // Raise error if any status other than Unimplemented is returned.
            ThrowIfError(maybe_stats.status());

            std::map<std::string, int64_t> result;
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
              result["bytes_reservable_limit"] =
                  *maybe_stats->bytes_reservable_limit;
            }
            result["largest_free_block_bytes"] =
                maybe_stats->largest_free_block_bytes;
            if (maybe_stats->pool_bytes) {
              result["pool_bytes"] = *maybe_stats->pool_bytes;
            }
            if (maybe_stats->peak_pool_bytes) {
              result["peak_pool_bytes"] = *maybe_stats->peak_pool_bytes;
            }
            return result;
          },
          "Returns memory statistics for this device keyed by name. May not be "
          "implemented on all platforms, and different platforms may return "
          "different stats, or -1 for unavailable stats. 'bytes_in_use' is "
          "usually available. Intended for diagnostic use.")
      .def("get_stream_for_external_ready_events",
           xla::ValueOrThrowWrapper(
               &PjRtDevice::GetStreamForExternalReadyEvents));
  static PyMethodDef get_attr_method = {
      "__getattr__",
      +[](PyObject* self, PyObject* args) -> PyObject* {
        PyObject* key;
        if (!PyArg_ParseTuple(args, "O", &key)) {
          PyErr_SetString(PyExc_TypeError, "__getattr__ must take 1 argument.");
          return nullptr;
        }
        try {
          auto device = py::cast<PjRtDevice*>(py::handle(self));
          auto name = py::cast<std::string>(py::handle(key));
          const auto& attrs = device->Attributes();
          auto it = attrs.find(name);
          if (it != attrs.end()) {
            auto result =
                std::visit([](auto&& v) { return py::cast(v); }, it->second);
            return result.release().ptr();
          }
          PyErr_SetNone(PyExc_AttributeError);
          return nullptr;
        } catch (std::exception& e) {
          PyErr_Format(PyExc_SystemError,
                       "Some unhandled pybind11 exception: %s", e.what());
          return nullptr;
        } catch (...) {
          PyErr_SetString(PyExc_SystemError,
                          "Some unhandled pybind11 exception.");
          return nullptr;
        }
      },
      METH_VARARGS,
      nullptr,
  };
  device.attr("__getattr__") =
      py::reinterpret_steal<py::object>(PyDescr_NewMethod(
          reinterpret_cast<PyTypeObject*>(device.ptr()), &get_attr_method));

  py::class_<PjRtMemorySpace, ClientAndPtr<PjRtMemorySpace>> memory_space(
      m, "Memory");
  memory_space
      .def_property_readonly(
          "process_index",
          [](const ClientAndPtr<PjRtMemorySpace>& memory_space) {
            return memory_space.client()->process_index();
          })
      .def_property_readonly(
          "platform",
          [](const ClientAndPtr<PjRtMemorySpace>& memory_space) {
            // TODO(phawkins): this is a temporary backwards
            // compatibility shim. We changed the name PJRT
            // reports for GPU platforms to "cuda" or "rocm",
            // but we haven't yet updated JAX clients that
            // expect "gpu". Migrate users and remove this
            // code.
            if (memory_space.client()->platform_name() == "cuda" ||
                memory_space.client()->platform_name() == "rocm") {
              return absl::string_view("gpu");
            } else {
              return memory_space.client()->platform_name();
            }
          })
      .def_property_readonly("kind", &PjRtMemorySpace::memory_space_kind)
      .def("__str__", &PjRtMemorySpace::DebugString)
      .def("__repr__", &PjRtMemorySpace::ToString)
      // Returns the devices that can address this `Memory`.
      .def("addressable_by_devices",
           [](const ClientAndPtr<PjRtMemorySpace>& memory_space) {
             std::vector<ClientAndPtr<PjRtDevice>> devices;
             auto span = memory_space->devices();
             devices.reserve(span.size());
             for (PjRtDevice* device : span) {
               devices.push_back(WrapWithClient(memory_space.client(), device));
             }
             return devices;
           });

  py::class_<PjRtLayout>(m, "PjRtLayout")
      .def("__str__", &PjRtLayout::ToString)
      .def("__eq__", [](const PjRtLayout& layout,
                        const PjRtLayout& other) { return layout == other; })
      .def("__hash__",
           [](const PjRtLayout& layout) { return absl::HashOf(layout); })
      .def(py::pickle(
          [](const PjRtLayout& layout) -> py::tuple {
            StatusOr<std::string> serialized = layout.Serialize();
            ThrowIfError(serialized.status());
            return py::make_tuple(py::bytes(*serialized));
          },
          [](py::tuple t) {
            // TODO(b/328671718): don't assume PjRtXlaLayout. We probably want a
            // generic method on PjRtCompiler instead, although we'll have
            // somehow have to attach a compiler to this PjRtLayout (something
            // like ClientAndPtr).
            StatusOr<PjRtXlaLayout> layout =
                PjRtXlaLayout::Deserialize(t[0].cast<std::string>());
            ThrowIfError(layout.status());
            return std::unique_ptr<PjRtLayout>(
                new PjRtXlaLayout(std::move(*layout)));
          }));

  // Local XLA client methods.

  py::enum_<PjRtClient::HostBufferSemantics>(m, "HostBufferSemantics")
      .value("IMMUTABLE_ONLY_DURING_CALL",
             PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall)
      .value("IMMUTABLE_UNTIL_TRANSFER_COMPLETES",
             PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes)
      .value("ZERO_COPY", PjRtClient::HostBufferSemantics::kZeroCopy);

  jax::BuildWeakrefLRUCacheAPI(m_nb);

  py::class_<PyClient, std::shared_ptr<PyClient>> py_local_client(m, "Client");
  py_local_client.def_property_readonly("platform", &PyClient::platform_name)
      .def_property_readonly("platform_version", &PyClient::platform_version)
      .def_property_readonly("runtime_type", &PyClient::runtime_type)
      .def("device_count", &PyClient::device_count)
      .def("local_device_count", &PyClient::addressable_device_count)
      .def("devices", &PyClient::Devices)
      .def("local_devices", &PyClient::LocalDevices)
      .def("device_from_local_hardware_id",
           xla::ValueOrThrowWrapper(&PyClient::DeviceFromLocalHardwareId))
      // TODO(phawkins): revert to the following after nanobind transition is
      // complete
      // .def("live_executables", &PyClient::LiveExecutables)
      // .def("live_arrays", &PyClient::LiveArrays)
      // .def("live_buffers", &PyClient::LiveArrays)
      .def("live_executables",
           [](PyClient& client) {
             return py::reinterpret_steal<py::object>(
                 nb::cast(client.LiveExecutables()).release().ptr());
           })
      .def("live_arrays",
           [](const PyClient& client) {
             return py::reinterpret_steal<py::object>(
                 nb::cast(client.LiveArrays()).release().ptr());
           })
      .def("live_buffers",
           [](const PyClient& client) {
             return py::reinterpret_steal<py::object>(
                 nb::cast(client.LiveArrays()).release().ptr());
           })
      .def("process_index", &PyClient::process_index)
      .def("host_id", &PyClient::process_index)
      .def("task_id", &PyClient::process_index)
      .def(
          "buffer_from_pyval",
          [](py::handle py_client, py::handle argument, py::handle py_device,
             bool force_copy,
             PjRtClient::HostBufferSemantics host_buffer_semantics) {
            PyClient* client = fast_cast<PyClient>(py_client);
            PjRtDevice* device = py_device.is_none()
                                     ? nullptr
                                     : fast_cast<PjRtDevice>(py_device);
            return ValueOrThrow(client->BufferFromPyval(
                argument, device, force_copy, host_buffer_semantics));
          },
          py::arg("argument"), py::arg("device") = nullptr,
          py::arg("force_copy") = false,
          py::arg("host_buffer_semantics") =
              PjRtClient::HostBufferSemantics::kZeroCopy)
      .def("make_cross_host_receive_buffers",
           xla::ValueOrThrowWrapper(&PyClient::MakeCrossHostReceiveBuffers),
           py::arg("shapes"), py::arg("device"))
      .def(
          "compile",
          [](PyClient& self, std::string mlir_module, py::object options_py,
             std::vector<pybind11::capsule> host_callbacks) {
            // TODO(phawkins): just wrap PyClient::Compile directly when the
            // nanobind transition is complete.
            CompileOptions options;
            if (!options_py.is_none()) {
              try {
                options =
                    nb::cast<CompileOptions>(nb::handle(options_py.ptr()));
              } catch (std::exception& e) {
                throw py::type_error(e.what());
              }
            }
            return py::reinterpret_steal<py::object>(
                nb::cast(ValueOrThrow(self.Compile(mlir_module, options,
                                                   host_callbacks)))
                    .release()
                    .ptr());
          },
          py::arg("computation"), py::arg("compile_options") = py::none(),
          py::arg("host_callbacks") = std::vector<py::capsule>())
      .def("serialize_executable",
           // TODO(phawkins): revert to the following after nanobind transition
           // xla::ValueOrThrowWrapper(&PyClient::SerializeExecutable))
           [](const PyClient& self, py::object executable_py) {
             const PyLoadedExecutable* executable =
                 nb::cast<const PyLoadedExecutable*>(
                     nb::handle(executable_py.ptr()));
             return xla::ValueOrThrow(self.SerializeExecutable(*executable));
           })
      .def(
          "deserialize_executable",
          // TODO(phawkins): revert to the following after nanobind transition
          // is complete
          // xla::ValueOrThrowWrapper(&PyClient::DeserializeExecutable),
          [](PyClient& self, const std::string& serialized,
             py::object options_py,
             std::vector<pybind11::capsule> host_callbacks) {
            std::optional<CompileOptions> options;
            if (!options_py.is_none()) {
              try {
                options =
                    nb::cast<CompileOptions>(nb::handle(options_py.ptr()));
              } catch (std::exception& e) {
                throw py::type_error(e.what());
              }
            }
            auto out = nb::cast(xla::ValueOrThrow(self.DeserializeExecutable(
                serialized, options, host_callbacks)));
            return py::reinterpret_steal<py::object>(out.release().ptr());
          },
          py::arg("serialized"), py::arg("compile_options") = py::none(),
          py::arg("host_callbacks") = std::vector<py::capsule>())
      .def("heap_profile", xla::ValueOrThrowWrapper(&PyClient::HeapProfile))
      // TODO(zhangqiaorjc): Experimental.
      .def("defragment",
           [](PyClient& self) { xla::ThrowIfError(self.Defragment()); })
      .def("get_emit_python_callback_descriptor",
           xla::ValueOrThrowWrapper(&PyClient::GetEmitPythonCallbackDescriptor),
           py::arg("callable"), py::arg("operand_shapes"),
           py::arg("result_shapes") = py::none())
      .def(
          "make_python_callback_from_host_send_and_recv",
          // TODO(phawkins): revert to
          //  xla::ValueOrThrowWrapper(
          //      &PyClient::MakePythonCallbackUsingHostSendAndRecv),
          // when the nanobind transition is done.
          [](PyClient& self, py::function callable, py::object operand_shapes,
             py::object result_shapes,
             absl::Span<uint16_t const> send_channel_ids,
             absl::Span<uint16_t const> recv_channel_ids,
             py::function serializer) {
            return ValueOrThrow(self.MakePythonCallbackUsingHostSendAndRecv(
                callable,
                nb::cast<std::vector<Shape>>(nb::handle(operand_shapes.ptr())),
                nb::cast<std::vector<Shape>>(nb::handle(result_shapes.ptr())),
                send_channel_ids, recv_channel_ids, serializer));
          },

          py::arg("callable"), py::arg("operand_shapes"),
          py::arg("result_shapes"), py::arg("send_channel_ids"),
          py::arg("recv_channel_ids"), py::arg("serializer") = py::none())
      .def("__getattr__", [](PyClient& client, std::string name) -> py::object {
        const auto& attrs = client.attributes();
        auto it = attrs.find(name);
        if (it != attrs.end()) {
          return std::visit([](auto&& v) { return py::cast(v); }, it->second);
        }
        throw py::attribute_error(absl::StrCat("Unknown attribute ", name));
      });

  py::class_<xla::cpu::CollectivesInterface,
             std::shared_ptr<xla::cpu::CollectivesInterface>>
      cpu_collectives(m, "CpuCollectives");

  m.def(
      "make_gloo_tcp_collectives",
      [](std::shared_ptr<DistributedRuntimeClient> distributed_client,

         std::optional<std::string> hostname,
         std::optional<std::string> interface)
          -> std::shared_ptr<xla::cpu::CollectivesInterface> {
#ifdef __linux__
        std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;
        if (distributed_client != nullptr) {
          kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"cpu:");
        }
        auto gloo_kv_store = std::make_unique<cpu::GlooKeyValueStore>(kv_store);
        auto tcp_attrs = gloo::transport::tcp::attr();
        if (hostname) {
          tcp_attrs.hostname = *hostname;
        }
        if (interface) {
          tcp_attrs.iface = *interface;
        }
        auto tcp_device = gloo::transport::tcp::CreateDevice(tcp_attrs);
        return std::make_shared<cpu::GlooCollectives>(std::move(gloo_kv_store),
                                                      std::move(tcp_device));
#else   // __linux__
        throw xla::XlaRuntimeError(
            "make_gloo_tcp_collectives only implemented for linux");
#endif  // __linux__
      },
      py::arg("distributed_client"), py::arg("hostname") = std::nullopt,
      py::arg("interface") = std::nullopt);

  m.def(
      "get_tfrt_cpu_client",
      [](bool asynchronous,
         std::shared_ptr<DistributedRuntimeClient> distributed_client,
         int node_id, int num_nodes,
         std::shared_ptr<xla::cpu::CollectivesInterface> collectives)
          -> std::shared_ptr<PyClient> {
        py::gil_scoped_release gil_release;
        CpuClientOptions options;
        if (distributed_client != nullptr) {
          options.kv_store = GetDistributedKeyValueStore(distributed_client,
                                                         /*key_prefix=*/"cpu:");
          options.node_id = node_id;
          options.num_nodes = num_nodes;

          options.collectives = std::move(collectives);
        }

        options.asynchronous = asynchronous;
        std::unique_ptr<PjRtClient> client =
            xla::ValueOrThrow(GetTfrtCpuClient(options));
        return std::make_shared<PyClient>(
            ifrt::PjRtClient::Create(std::move(client)));
      },
      py::arg("asynchronous") = true, py::arg("distributed_client") = nullptr,
      py::arg("node_id") = 0, py::arg("num_nodes") = 1,
      py::arg("collectives") =
          std::shared_ptr<xla::cpu::CollectivesInterface>());
  m.def("pjrt_plugin_loaded", [](std::string platform_name) -> bool {
    xla::StatusOr<const PJRT_Api*> pjrt_api = pjrt::PjrtApi(platform_name);
    return pjrt_api.ok();
  });
  m.def(
      "load_pjrt_plugin",
      [](std::string platform_name, std::optional<std::string> library_path,
         std::optional<py::capsule> c_api) -> py::capsule {
        if (library_path.has_value()) {
          const PJRT_Api* api = xla::ValueOrThrow(
              pjrt::LoadPjrtPlugin(platform_name, *library_path));
          return py::capsule(absl::bit_cast<void*>(api), "pjrt_c_api");
        }
        if (absl::string_view(c_api->name()) != "pjrt_c_api") {
          throw py::value_error(
              "c_api argument to load_pjrt_plugin is not a pjrt_c_api "
              "capsule.");
        }
        xla::ThrowIfError(pjrt::SetPjrtApi(
            platform_name, static_cast<const PJRT_Api*>(*c_api)));
        return *c_api;
      },
      py::arg("platform_name"), py::arg("library_path") = std::nullopt,
      py::arg("c_api") = std::nullopt);
  m.def("pjrt_plugin_initialized", [](std::string platform_name) -> bool {
    return xla::ValueOrThrow(pjrt::IsPjrtPluginInitialized(platform_name));
  });
  m.def("initialize_pjrt_plugin", [](std::string platform_name) {
    return xla::ThrowIfError(pjrt::InitializePjrtPlugin(platform_name));
  });

#ifdef XLA_PYTHON_ENABLE_GPU
  py::class_<GpuAllocatorConfig> alloc_config(m, "GpuAllocatorConfig");
  alloc_config.def(py::init<>())
      .def_readwrite("kind", &GpuAllocatorConfig::kind)
      .def_readwrite("memory_fraction", &GpuAllocatorConfig::memory_fraction)
      .def_readwrite("preallocate", &GpuAllocatorConfig::preallocate)
      .def_readwrite("collective_memory_size",
                     &GpuAllocatorConfig::collective_memory_size);
  py::enum_<GpuAllocatorConfig::Kind>(alloc_config, "Kind")
      .value("DEFAULT", GpuAllocatorConfig::Kind::kDefault)
      .value("PLATFORM", GpuAllocatorConfig::Kind::kPlatform)
      .value("BFC", GpuAllocatorConfig::Kind::kBFC)
      .value("CUDA_ASYNC", GpuAllocatorConfig::Kind::kCudaAsync);

  m.def(
      "get_gpu_client",
      [](bool asynchronous, const GpuAllocatorConfig& allocator_config,
         std::shared_ptr<DistributedRuntimeClient> distributed_client,
         int node_id, int num_nodes,
         std::optional<std::set<int>> allowed_devices,
         std::optional<std::string> platform_name,
         std::optional<bool> mock = false) -> std::shared_ptr<PyClient> {
        py::gil_scoped_release gil_release;
        std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;
        if (distributed_client != nullptr) {
          kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"gpu:");
        }
        GpuClientOptions options;
        options.allocator_config = allocator_config;
        options.node_id = node_id;
        options.num_nodes = num_nodes;
        options.allowed_devices = allowed_devices;
        options.platform_name = platform_name;
        options.kv_store = kv_store;
        options.enable_mock_nccl = mock.value_or(false);
        std::unique_ptr<PjRtClient> client =
            xla::ValueOrThrow(GetStreamExecutorGpuClient(options));
        return std::make_shared<PyClient>(
            ifrt::PjRtClient::Create(std::move(client)));
      },
      py::arg("asynchronous") = true,
      py::arg("allocator_config") = GpuAllocatorConfig(),
      py::arg("distributed_client") = nullptr, py::arg("node_id") = 0,
      py::arg("num_nodes") = 1, py::arg("allowed_devices") = std::nullopt,
      py::arg("platform_name") = std::nullopt, py::arg("mock") = std::nullopt);
#endif  // XLA_PYTHON_ENABLE_GPU

  m.def(
      "get_c_api_client",
      [](std::string platform_name,
         const absl::flat_hash_map<std::string, PjRtValueType>& options,
         std::shared_ptr<DistributedRuntimeClient> distributed_client)
          -> std::shared_ptr<PyClient> {
        py::gil_scoped_release gil_release;
        std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;
        if (distributed_client != nullptr) {
          kv_store = GetDistributedKeyValueStore(
              distributed_client,
              /*key_prefix=*/absl::StrCat(platform_name, ":"));
        }
        std::unique_ptr<PjRtClient> c_api_client =
            xla::ValueOrThrow(GetCApiClient(platform_name, options, kv_store));
        return std::make_shared<PyClient>(
            ifrt::PjRtClient::Create(std::move(c_api_client)));
      },
      py::arg("platform_name"),
      py::arg("options") = absl::flat_hash_map<std::string, PjRtValueType>(),
      py::arg("distributed_client") = nullptr);
  // TODO(b/322357665): Delete this method after TPU plugin changes to use the
  // standard registration.
  m.def("get_default_c_api_topology",
        [](std::string platform_name, std::string topology_name,
           const absl::flat_hash_map<std::string, PjRtValueType>& options)
            -> std::shared_ptr<PjRtTopologyDescription> {
          return xla::ValueOrThrow(
              GetCApiTopology(platform_name, topology_name, options));
        });
  m.def("get_c_api_topology",
        [](py::capsule c_api, std::string topology_name,
           const absl::flat_hash_map<std::string, PjRtValueType>& options)
            -> std::shared_ptr<PjRtTopologyDescription> {
          if (absl::string_view(c_api.name()) != "pjrt_c_api") {
            throw py::value_error(
                "Argument to get_c_api_topology was not a pjrt_c_api capsule.");
          }
          return xla::ValueOrThrow(GetCApiTopology(
              static_cast<const PJRT_Api*>(c_api), topology_name, options));
        });
  m.def("get_topology_for_devices",
        [](std::vector<ClientAndPtr<PjRtDevice>> devices_and_clients) {
          if (devices_and_clients.empty()) {
            throw py::value_error(
                "get_topology_for_devices requires >= 1 devices.");
          }
          auto client = devices_and_clients[0].client();
          std::vector<PjRtDevice*> devices;
          devices.reserve(devices_and_clients.size());
          for (const ClientAndPtr<PjRtDevice>& device : devices_and_clients) {
            if (device.get_client() != client.get()) {
              throw py::value_error(
                  "devices passed to get_topology_for_devices come from "
                  "different clients.");
            }
            devices.push_back(device.get());
          }
          return xla::ValueOrThrow(client->ifrt_client()->GetTopologyForDevices(
              absl::MakeSpan(devices)));
        });

  TF_CHECK_OK(PyArray::RegisterTypes(m_nb));
  jax::RegisterDeviceList(m_nb);
  jax::RegisterSharding(m_nb);

  nb::class_<CompiledMemoryStats>(m_nb, "CompiledMemoryStats")
      .def_rw("generated_code_size_in_bytes",
              &CompiledMemoryStats::generated_code_size_in_bytes)
      .def_rw("argument_size_in_bytes",
              &CompiledMemoryStats::argument_size_in_bytes)
      .def_rw("output_size_in_bytes",
              &CompiledMemoryStats::output_size_in_bytes)
      .def_rw("alias_size_in_bytes", &CompiledMemoryStats::alias_size_in_bytes)
      .def_rw("temp_size_in_bytes", &CompiledMemoryStats::temp_size_in_bytes)
      .def_rw("host_generated_code_size_in_bytes",
              &CompiledMemoryStats::host_generated_code_size_in_bytes)
      .def_rw("host_argument_size_in_bytes",
              &CompiledMemoryStats::host_argument_size_in_bytes)
      .def_rw("host_output_size_in_bytes",
              &CompiledMemoryStats::host_output_size_in_bytes)
      .def_rw("host_alias_size_in_bytes",
              &CompiledMemoryStats::host_alias_size_in_bytes)
      .def_rw("host_temp_size_in_bytes",
              &CompiledMemoryStats::host_temp_size_in_bytes)
      .def_prop_ro("serialized_hlo_proto",
                   [](const CompiledMemoryStats& cms) -> nb::bytes {
                     return nb::bytes(cms.serialized_hlo_proto.data(),
                                      cms.serialized_hlo_proto.size());
                   })
      .def("__str__", &CompiledMemoryStats::DebugString);

  nb::class_<PyExecuteResults>(m_nb, "ExecuteResults")
      .def("__len__", [](PyExecuteResults& results) { return results.Size(); })
      .def("disassemble_into_single_device_arrays",
           &PyExecuteResults::DisassembleIntoSingleDeviceArrays)
      .def("disassemble_prefix_into_single_device_arrays",
           &PyExecuteResults::DisassemblePrefixIntoSingleDeviceArrays)
      .def("consume_with_handlers", &PyExecuteResults::ConsumeWithHandlers)
      .def("consume_token", &PyExecuteResults::ConsumeToken);

  nb::class_<PyLoadedExecutable>(m_nb, "LoadedExecutable")
      .def_prop_ro(
          "client",
          // TODO(phawkins): directly wrap method after nanobind transition.
          [](const PyLoadedExecutable& self) -> nb::object {
            return nb::borrow(py::cast(self.client()).ptr());
          })
      .def("local_logical_device_ids",
           [](PyLoadedExecutable* exec) {
             auto span = exec->addressable_device_logical_ids();
             // Not on dispatch critical path, so ok to have heap allocation.
             std::vector<std::pair<int, int>> addressable_device_logic_ids;
             addressable_device_logic_ids.reserve(span.size());
             for (const auto& logical_device_id : span) {
               addressable_device_logic_ids.push_back(std::make_pair(
                   logical_device_id.replica, logical_device_id.partition));
             }
           })
      // TODO(phawkins): directly wrap after nanobind transition
      // .def("local_devices", &PyLoadedExecutable::AddressableDevices)
      .def("local_devices",
           [](const PyLoadedExecutable& self) {
             return nb::borrow(py::cast(self.AddressableDevices()).ptr());
           })
      .def("size_of_generated_code_in_bytes",
           &PyLoadedExecutable::SizeOfGeneratedCodeInBytes)
      .def(
          "get_compiled_memory_stats",
          xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetCompiledMemoryStats))
      .def("delete", &PyLoadedExecutable::Delete)
      .def("execute_sharded_on_local_devices",
           xla::ValueOrThrowWrapper(
               &PyLoadedExecutable::ExecuteShardedOnLocalDevices),
           nb::arg("arguments"))
      .def("execute_sharded_on_local_devices_with_tokens",
           xla::ValueOrThrowWrapper(
               &PyLoadedExecutable::ExecuteShardedOnLocalDevicesWithTokens),
           nb::arg("arguments"))
      // TODO(parkers): Switch execute_sharded_on_local_devices* to this.
      .def("execute_sharded",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::ExecuteSharded),
           nb::arg("arguments"), nb::arg("with_tokens") = false)
      .def("hlo_modules", ValueOrThrowWrapper(&PyLoadedExecutable::HloModules))
      .def("get_output_memory_kinds",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetOutputMemoryKinds))
      .def("get_output_shardings", &PyLoadedExecutable::GetOutputShardings)
      .def("get_parameter_layouts",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetParameterLayouts))
      .def("get_output_layouts",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetOutputLayouts))
      .def("get_parameter_shardings",
           &PyLoadedExecutable::GetParameterShardings)
      .def("keep_alive", &PyLoadedExecutable::KeepAlive)
      .def("compile_options",
           [](const PyLoadedExecutable& self) {
             return xla::ValueOrThrow(
                 self.pjrt_executable()->GetCompileOptions());
           })
      .def("cost_analysis",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetCostAnalysis))
      .def_prop_ro("traceback", &PyLoadedExecutable::traceback)
      .def_prop_ro("fingerprint", [](PyLoadedExecutable* exec) -> nb::object {
        if (exec->fingerprint().has_value()) {
          return nb::bytes(exec->fingerprint()->data(),
                           exec->fingerprint()->size());
        } else {
          return nb::none();
        }
      });
  nb::class_<PyToken> token(m_nb, "Token");
  token.def("block_until_ready",
            [](PyToken& self) { xla::ThrowIfError(self.Await()); });

  nb::class_<PyShardedToken> sharded_token(m_nb, "ShardedToken");
  sharded_token.def("block_until_ready", [](PyShardedToken& self) {
    xla::ThrowIfError(self.Await());
  });
  sharded_token.def("get_token", &PyShardedToken::GetPyToken);

  m.def("buffer_to_dlpack_managed_tensor",
        xla::ValueOrThrowWrapper(BufferToDLPackManagedTensor),
        py::arg("buffer"), py::arg("stream") = py::none());
  m.def("dlpack_managed_tensor_to_buffer",
        [](const pybind11::capsule& tensor, ClientAndPtr<PjRtDevice> device,
           std::optional<std::intptr_t> stream) {
          return xla::ValueOrThrow(DLPackManagedTensorToBuffer(
              tensor, device.get(), device.client(), stream));
        });
  // Legacy overload
  m.def(
      "dlpack_managed_tensor_to_buffer",
      [](const pybind11::capsule& tensor, std::shared_ptr<PyClient> cpu_client,
         std::shared_ptr<PyClient> gpu_client) {
        return xla::ValueOrThrow(DLPackManagedTensorToBuffer(
            tensor, std::move(cpu_client), std::move(gpu_client)));
      },
      py::arg("dlpack"), py::arg("cpu_backend") = nullptr,
      py::arg("gpu_backend") = nullptr);
  m.def("cuda_array_interface_to_buffer",
        [](py::handle cai_py, std::shared_ptr<PyClient> cuda_client) {
          // TODO(phawkins): simplify after nanobind transition is complete.
          nb::dict cai = nb::cast<nb::dict>(nb::handle(cai_py.ptr()));
          auto out = xla::ValueOrThrow(
              CudaArrayInterfaceToBuffer(cai, std::move(cuda_client)));
          return py::reinterpret_steal<py::object>(out.release().ptr());
        });
  BuildProfilerSubmodule(m_nb);
  BuildOpsSubmodule(m_nb);
  BuildOutfeedReceiverSubmodule(m_nb);
  BuildPytreeSubmodule(m_nb);
  jax::BuildJaxjitSubmodule(m_nb);
  jax::BuildPmapSubmodule(m_nb);
  jax::BuildPjitSubmodule(m_nb);
  jax::BuildTransferGuardSubmodule(m_nb);
  BuildTracebackSubmodule(m_nb);
  BuildMlirSubmodule(m_nb);
  BuildCustomCallShardingPybindAPI(m_nb);

  py::class_<tsl::PreemptionSyncManager,
             std::unique_ptr<tsl::PreemptionSyncManager>>
      preemption_sync_manager(m, "PreemptionSyncManager");
  preemption_sync_manager
      .def(
          "initialize",
          [](tsl::PreemptionSyncManager& manager,
             DistributedRuntimeClient* client) {
            tsl::CoordinationServiceAgent* agent =
                xla::ValueOrThrow(client->GetCoordinationServiceAgent());
            xla::ThrowIfError(manager.Initialize(agent));
          },
          py::arg("distributed_client"))
      .def("reached_sync_point",
           [](tsl::PreemptionSyncManager& manager, int step_counter) {
             return manager.ReachedSyncPoint(step_counter);
           });
  m.def("create_preemption_sync_manager",
        []() { return tsl::CreatePreemptionSyncManager(); });

  py::class_<DistributedRuntimeService,
             std::unique_ptr<DistributedRuntimeService>>
      distributed_runtime_service(m, "DistributedRuntimeService");
  distributed_runtime_service.def("shutdown",
                                  &DistributedRuntimeService::Shutdown,
                                  py::call_guard<py::gil_scoped_release>());
  py::class_<DistributedRuntimeClient,
             std::shared_ptr<DistributedRuntimeClient>>
      distributed_runtime_client(m, "DistributedRuntimeClient");
  distributed_runtime_client
      .def("connect",
           [](DistributedRuntimeClient& self) {
             py::gil_scoped_release gil_release;
             xla::ThrowIfError(self.Connect());
           })
      .def("shutdown",
           [](DistributedRuntimeClient& self) {
             py::gil_scoped_release gil_release;
             xla::ThrowIfError(self.Shutdown());
           })
      // This method assumes that the value is a Python string. Use
      // `blocking_key_value_get_bytes()` if key_value_set() was called with a
      // Python bytes object as its value.
      .def(
          "blocking_key_value_get",
          [](DistributedRuntimeClient& client, std::string key,
             int64_t timeout_in_ms) {
            py::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.BlockingKeyValueGet(
                key, absl::Milliseconds(timeout_in_ms)));
          },
          py::arg("key"), py::arg("timeout_in_ms"))
      // Same as `blocking_key_value_get()`, but retrieves the raw Python byte
      // values explicitly.
      .def(
          "blocking_key_value_get_bytes",
          [](DistributedRuntimeClient& client, std::string key,
             int64_t timeout_in_ms) -> py::bytes {
            py::gil_scoped_release gil_release;
            std::string result = xla::ValueOrThrow(client.BlockingKeyValueGet(
                key, absl::Milliseconds(timeout_in_ms)));
            return py::bytes(result);
          },
          py::arg("key"), py::arg("timeout_in_ms"))
      .def(
          "wait_at_barrier",
          [](DistributedRuntimeClient& client, std::string barrier_id,
             int64_t timeout_in_ms) {
            py::gil_scoped_release gil_release;
            xla::ThrowIfError(client.WaitAtBarrier(
                barrier_id, absl::Milliseconds(timeout_in_ms)));
          },
          py::arg("barrier_id"), py::arg("timeout_in_ms"))
      // The key must be a string, but the value can either be a Python string
      // or bytes object.
      // With Python string values, use `key_value_set()` and
      // `blocking_key_value_get()`.
      // With Python byte object values, use `key_value_set()` and
      // `blocking_key_value_get_bytes()`.
      .def(
          "key_value_set",
          [](DistributedRuntimeClient& client, std::string key,
             std::string value) {
            py::gil_scoped_release gil_release;
            xla::ThrowIfError(client.KeyValueSet(key, value));
          },
          py::arg("key"), py::arg("value"))
      // The key must be a string, but the value must a Python bytes object.
      // Use `key_value_set_bytes()` and `blocking_key_value_get_bytes()`.
      .def(
          "key_value_set_bytes",
          [](DistributedRuntimeClient& client, std::string key,
             py::bytes value) {
            py::gil_scoped_release gil_release;
            xla::ThrowIfError(client.KeyValueSet(key, value));
          },
          py::arg("key"), py::arg("value"))
      // Assumes that all values in the directory are Python strings.
      .def(
          "key_value_dir_get",
          [](DistributedRuntimeClient& client, std::string key) {
            py::gil_scoped_release gil_release;
            return xla::ValueOrThrow(client.KeyValueDirGet(key));
          },
          py::arg("key"))
      // Assumes that all values in the directory are Python byte objects.
      // Same as `key_value_dir_get()`, but retrieves Python byte values
      // explicitly.
      .def(
          "key_value_dir_get_bytes",
          [](DistributedRuntimeClient& client, std::string key)
              -> std::vector<std::pair<std::string, py::bytes>> {
            py::gil_scoped_release gil_release;
            std::vector<std::pair<std::string, std::string>> result =
                xla::ValueOrThrow(client.KeyValueDirGet(key));
            // Convert std::string values to py::bytes.
            std::vector<std::pair<std::string, py::bytes>> kvs;
            kvs.reserve(result.size());
            for (const auto& kv : result) {
              kvs.push_back(std::pair(kv.first, py::bytes(kv.second)));
            }
            return kvs;
          },
          py::arg("key"))
      .def(
          "key_value_delete",
          [](DistributedRuntimeClient& client, std::string key) {
            py::gil_scoped_release gil_release;
            return client.KeyValueDelete(key);
          },
          py::arg("key"));

  m.def(
      "get_distributed_runtime_service",
      [](std::string address, int num_nodes,
         std::optional<int> heartbeat_interval,
         std::optional<int> max_missing_heartbeats,
         std::optional<int> cluster_register_timeout,
         std::optional<int> shutdown_timeout)
          -> std::unique_ptr<DistributedRuntimeService> {
        CoordinationServiceImpl::Options options;
        options.num_nodes = num_nodes;
        if (heartbeat_interval.has_value()) {
          options.heartbeat_interval = absl::Seconds(*heartbeat_interval);
        }
        if (max_missing_heartbeats.has_value()) {
          options.max_missing_heartbeats = *max_missing_heartbeats;
        }
        if (cluster_register_timeout.has_value()) {
          options.cluster_register_timeout =
              absl::Seconds(*cluster_register_timeout);
        }
        if (shutdown_timeout.has_value()) {
          options.shutdown_timeout = absl::Seconds(*shutdown_timeout);
        }
        std::unique_ptr<DistributedRuntimeService> service =
            xla::ValueOrThrow(GetDistributedRuntimeService(address, options));
        return service;
      },
      py::arg("address"), py::arg("num_nodes"), py::kw_only(),
      py::arg("heartbeat_interval") = std::nullopt,
      py::arg("max_missing_heartbeats") = std::nullopt,
      py::arg("cluster_register_timeout") = std::nullopt,
      py::arg("shutdown_timeout") = std::nullopt);

  m.def(
      "get_distributed_runtime_client",
      [](std::string address, int node_id, std::optional<int> rpc_timeout,
         std::optional<int> init_timeout, std::optional<int> shutdown_timeout,
         std::optional<int> heartbeat_interval,
         std::optional<int> max_missing_heartbeats,
         std::optional<std::function<void(xla::Status,
                                          bool coordinator_reported_failure)>>
             missed_heartbeat_callback,
         std::optional<bool> shutdown_on_destruction)
          -> std::shared_ptr<DistributedRuntimeClient> {
        DistributedRuntimeClient::Options options;
        options.node_id = node_id;
        if (rpc_timeout.has_value()) {
          options.rpc_timeout = absl::Seconds(*rpc_timeout);
        }
        if (init_timeout.has_value()) {
          options.init_timeout = absl::Seconds(*init_timeout);
        }
        if (shutdown_timeout.has_value()) {
          options.shutdown_timeout = absl::Seconds(*shutdown_timeout);
        }
        if (heartbeat_interval.has_value()) {
          options.heartbeat_interval = absl::Seconds(*heartbeat_interval);
        }
        if (max_missing_heartbeats.has_value()) {
          options.max_missing_heartbeats = *max_missing_heartbeats;
        }
        if (missed_heartbeat_callback.has_value()) {
          options.missed_heartbeat_callback =
              std::move(*missed_heartbeat_callback);
        }
        if (shutdown_on_destruction.has_value()) {
          options.shutdown_on_destruction = *shutdown_on_destruction;
        }
        return GetDistributedRuntimeClient(address, options);
      },
      py::arg("address"), py::arg("node_id"), py::kw_only(),
      py::arg("rpc_timeout") = std::nullopt,
      py::arg("init_timeout") = std::nullopt,
      py::arg("shutdown_timeout") = std::nullopt,
      py::arg("heartbeat_interval") = std::nullopt,
      py::arg("max_missing_heartbeats") = std::nullopt,
      py::arg("missed_heartbeat_callback") = std::nullopt,
      py::arg("shutdown_on_destruction") = std::nullopt);

  m.def("collect_garbage", []() { GlobalPyRefManager()->CollectGarbage(); });

  m.def("is_optimized_build", &IsOptimizedBuild);

  m_nb.def("json_to_pprof_profile",
           xla::ValueOrThrowWrapper(JsonToPprofProfile),
           "Encodes the JSON representation of a pprof Profile into its binary "
           "protocol buffer encoding.");
  m_nb.def("pprof_profile_to_json",
           xla::ValueOrThrowWrapper(PprofProfileToJson),
           "Decodes an uncompressed pprof Profile protocol buffer into a JSON "
           "representation");

  RegisterCompileOnlyClient(m);
  py::class_<PjRtTopologyDescription, std::shared_ptr<PjRtTopologyDescription>>(
      m, "DeviceTopology")
      .def("_make_compile_only_devices",
           [](std::shared_ptr<PjRtTopologyDescription> topology) {
             return MakeCompileOnlyClient(topology)->Devices();
           })
      .def_property_readonly("platform",
                             [](PjRtTopologyDescription& topology) {
                               return topology.platform_name();
                             })
      .def_property_readonly("platform_version",
                             [](PjRtTopologyDescription& topology) {
                               return topology.platform_version();
                             })
      .def("serialize",
           [](PjRtTopologyDescription& topology) -> py::bytes {
             return py::bytes(ValueOrThrow(topology.Serialize()));
           })
      .def(
          "__getattr__",
          [](PjRtTopologyDescription& topology,
             std::string name) -> py::object {
            const auto& attrs = topology.Attributes();
            auto it = attrs.find(name);
            if (it != attrs.end()) {
              return std::visit([](auto&& v) { return py::cast(v); },
                                it->second);
            }
            throw py::attribute_error(absl::StrCat("Unknown attribute ", name));
          });

  py::class_<PjRtExecutable, std::shared_ptr<PjRtExecutable>>(m, "Executable")
      .def("hlo_modules",
           [](const PjRtExecutable& self) {
             // TODO(phawkins): revert to a direct wrapping of
             // PyLoadedExecutable::GetParameterLayouts when nanobind transition
             // is complete.
             return py::reinterpret_steal<py::object>(
                 nb::cast(ValueOrThrow(self.GetHloModules())).release().ptr());
           })
      .def("get_output_memory_kinds",
           xla::ValueOrThrowWrapper(&PjRtExecutable::GetOutputMemoryKinds))
      .def("get_output_shardings",
           [](const PjRtExecutable& self) {
             return py::reinterpret_borrow<py::object>(
                 nb::cast(self.GetOutputShardings()).release().ptr());
           })
      .def("get_parameter_layouts",
           [](const PjRtExecutable& self) {
             // TODO(phawkins): revert to a direct wrapping of
             // PjRtExecutable::GetParameterLayouts when nanobind transition
             // is complete.
             // xla::ValueOrThrowWrapper(&PjRtExecutable::GetParameterLayouts)
             return py::reinterpret_steal<py::object>(
                 nb::cast(ValueOrThrow(self.GetParameterLayouts()))
                     .release()
                     .ptr());
           })
      .def("get_output_layouts",
           [](const PjRtExecutable& self) {
             // TODO(phawkins): revert to a direct wrapping of
             // PjRtExecutable::GetOutputLayouts when nanobind transition
             // is complete.
             // xla::ValueOrThrowWrapper(&PjRtExecutable::GetOutputLayouts)
             return py::reinterpret_steal<py::object>(
                 nb::cast(ValueOrThrow(self.GetOutputLayouts()))
                     .release()
                     .ptr());
           })
      .def("get_parameter_shardings",
           [](const PjRtExecutable& self) {
             return py::reinterpret_borrow<py::object>(
                 nb::cast(self.GetParameterShardings()).release().ptr());
           })
      .def("get_compiled_memory_stats",
           [](const PjRtExecutable& self) {
             // TODO(phawkins): revert to a direct wrapping of
             // PjRtExecutable::GetCompiledMemoryStats when nanobind transition
             // is complete.
             // xla::ValueOrThrowWrapper(&PjRtExecutable::GetCompiledMemoryStats)
             return py::reinterpret_steal<py::object>(
                 nb::cast(ValueOrThrow(self.GetCompiledMemoryStats()))
                     .release()
                     .ptr());
           })
      .def("compile_options",
           // TODO(phawkins): revert to the following when nanobind transition
           // complete
           // xla::ValueOrThrowWrapper(&PjRtExecutable::GetCompileOptions))
           [](const PjRtExecutable& self) {
             return py::reinterpret_steal<py::object>(
                 nb::cast(ValueOrThrow(self.GetCompileOptions()))
                     .release()
                     .ptr());
           })
      .def("serialize",
           [](const PjRtExecutable& exec) -> py::bytes {
             return ValueOrThrow(exec.SerializeExecutable());
           })
      .def("cost_analysis",
           xla::ValueOrThrowWrapper(&PjRtExecutable::GetCostAnalysis));

  m.def("is_asan", IsAsan);
  m.def("is_msan", IsMsan);
  m.def("is_tsan", IsTsan);
  m.def("is_sanitized", IsSanitized);

  m.def(
      "batched_device_put",
      [](py::object aval, py::object sharding, py::object xs_py,
         std::vector<ClientAndPtr<PjRtDevice>> dst_devices, bool committed,
         bool force_copy,
         PjRtClient::HostBufferSemantics host_buffer_semantics) -> py::object {
        // TODO(phawkins): simplify after nanobind transition is complete.
        auto xs = nb::cast<std::vector<nb::object>>(nb::handle(xs_py.ptr()));
        return py::reinterpret_steal<py::object>(
            ValueOrThrow(PyArray::BatchedDevicePut(
                             nb::borrow(aval.ptr()), nb::borrow(sharding.ptr()),
                             std::move(xs), std::move(dst_devices), committed,
                             force_copy, host_buffer_semantics,
                             jax::GetEnableX64()))
                .release()
                .ptr());
      },
      py::arg("aval"), py::arg("sharding"), py::arg("xs"), py::arg("devices"),
      py::arg("committed") = true, py::arg("force_copy") = false,
      py::arg("host_buffer_semantics") =
          PjRtClient::HostBufferSemantics::kZeroCopy);
  m_nb.def("check_and_canonicalize_memory_kind",
           &jax::CheckAndCanonicalizeMemoryKind, nb::arg("memory_kind").none(),
           nb::arg("device_list"));
}  // NOLINT(readability/fn_size)

// This code in essence is a copy of PYBIND11_MODULE(). We can't just call
// PYBIND11_MODULE because we want the entry point of the module to be in
// the py_extension() translation unit but we don't want anything else to be
// defined there. Inside Google, py_extension() translation units are linked
// differently and they end up with a different instance of the
// py::module_local() state, breaking that feature of pybind11.
static py::module_::module_def xla_module_def;

PyObject* InitializeXlaExtension() {
  PYBIND11_CHECK_PYTHON_VERSION
  PYBIND11_ENSURE_INTERNALS_READY
  auto m = py::module_::create_extension_module("xla_extension", nullptr,
                                                &xla_module_def);
  try {
    Init(m);
    return m.ptr();
  }
  PYBIND11_CATCH_INIT_EXCEPTIONS
}

}  // namespace xla
