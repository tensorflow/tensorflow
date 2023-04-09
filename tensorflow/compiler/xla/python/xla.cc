/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// clang-format off
// Must be included first
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/tsl/python/lib/core/numpy.h"  //NOLINT
// clang-format on

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"
#include "tensorflow/compiler/xla/pjrt/distributed/service.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#ifdef XLA_PYTHON_ENABLE_GPU
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#endif  // XLA_PYTHON_ENABLE_GPU
#include "tensorflow/compiler/xla/pjrt/interpreter_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"
#ifdef XLA_PYTHON_ENABLE_PLUGIN_DEVICE
#include "tensorflow/compiler/xla/pjrt/pjrt_plugin_device_client.h"
#endif  // XLA_PYTHON_ENABLE_PLUGIN_DEVICE
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#ifdef XLA_PYTHON_ENABLE_TPU
#include "tensorflow/compiler/xla/pjrt/pjrt_c_api_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#endif  // XLA_PYTHON_ENABLE_TPU
#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"
#include "tensorflow/compiler/xla/python/custom_call_sharding.h"
#include "tensorflow/compiler/xla/python/dlpack.h"
#include "tensorflow/compiler/xla/python/jax_jit.h"
#include "tensorflow/compiler/xla/python/mlir.h"
#include "tensorflow/compiler/xla/python/ops.h"
#include "tensorflow/compiler/xla/python/outfeed_receiver_py.h"
#include "tensorflow/compiler/xla/python/pjit.h"
#include "tensorflow/compiler/xla/python/pmap_lib.h"
#include "tensorflow/compiler/xla/python/pprof_profile_builder.h"
#include "tensorflow/compiler/xla/python/profiler.h"
#include "tensorflow/compiler/xla/python/py_array.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/python/sharding.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/python/weakref_lru_cache.h"
#include "tensorflow/compiler/xla/python/xla_compiler.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/distributed_runtime/preemption/preemption_sync_manager.h"

// TODO(phawkins): remove host_id properties after JAX is update to avoid them.

namespace xla {
namespace {

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

PYBIND11_MODULE(xla_extension, m) {
  tsl::ImportNumpy();

  // Exceptions
  py::register_exception<XlaRuntimeError>(m, "XlaRuntimeError",
                                          PyExc_RuntimeError);

  // Types
  py::enum_<PrimitiveType>(m, "PrimitiveType")
      .value("PRIMITIVE_TYPE_INVALID", PRIMITIVE_TYPE_INVALID)
      .value("PRED", PRED)
      .value("S8", S8)
      .value("S16", S16)
      .value("S32", S32)
      .value("S64", S64)
      .value("U8", U8)
      .value("U16", U16)
      .value("U32", U32)
      .value("U64", U64)
      .value("F16", F16)
      .value("F8E4M3FN", F8E4M3FN)
      .value("F8E5M2", F8E5M2)
      .value("BF16", BF16)
      .value("F32", F32)
      .value("F64", F64)
      .value("C64", C64)
      .value("C128", C128)
      .value("TUPLE", TUPLE)
      .value("OPAQUE_TYPE", OPAQUE_TYPE)
      .value("TOKEN", TOKEN);

  // Must be before PyClient.compile.
  BuildXlaCompilerSubmodule(m);

  py::class_<PjRtDevice, ClientAndPtr<PjRtDevice>>(
      m, "Device",
      "A descriptor of an available device.\n\nSubclasses are used to "
      "represent specific types of devices, e.g. CPUs, GPUs. Subclasses may "
      "have additional properties specific to that device type.")
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
                               return device.client->platform_name();
                             })
      .def_property_readonly("device_kind", &PjRtDevice::device_kind)
      .def_property_readonly(
          "client",
          [](const ClientAndPtr<PjRtDevice>& device) { return device.client; })
      .def("__str__", &PjRtDevice::DebugString)
      .def("__repr__", &PjRtDevice::ToString)
      .def("transfer_to_infeed",
           [](PjRtDevice& device, const LiteralSlice& literal) {
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             xla::ThrowIfError(device.TransferToInfeed(literal));
           })
      .def("transfer_from_outfeed",
           [](PjRtDevice& device, const Shape& shape) -> py::object {
             GlobalPyRefManager()->CollectGarbage();
             std::shared_ptr<Literal> literal;
             {
               py::gil_scoped_release gil_release;
               Shape shape_with_layout = shape;
               ShapeUtil::ForEachMutableSubshape(
                   &shape_with_layout, [](Shape* subshape, const ShapeIndex&) {
                     if (!subshape->has_layout()) {
                       LayoutUtil::SetToDefaultLayout(subshape);
                     }
                   });
               literal = std::make_shared<Literal>(shape_with_layout);
               xla::ThrowIfError(device.TransferFromOutfeed(literal.get()));
             }
             return ValueOrThrow(LiteralToPython(std::move(literal)));
           })
      .def("live_buffers",
           [](const ClientAndPtr<PjRtDevice>& device) {
             PythonDeprecationWarning(
                 "Per device live_buffers() is going to be deprecated. Please "
                 "use the jax.live_arrays() for jax.Arrays instead.");
             return py::list();
           })
      .def(
          "__getattr__",
          [](PjRtDevice& device, std::string name) -> py::object {
            const auto& attrs = device.Attributes();
            auto it = attrs.find(name);
            if (it != attrs.end()) {
              return std::visit([](auto&& v) { return py::cast(v); },
                                it->second);
            }
            throw py::attribute_error(absl::StrCat("Unknown attribute ", name));
          });

  // Local XLA client methods.

  py::enum_<PjRtClient::HostBufferSemantics>(m, "HostBufferSemantics")
      .value("IMMUTABLE_ONLY_DURING_CALL",
             PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall)
      .value("IMMUTABLE_UNTIL_TRANSFER_COMPLETES",
             PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes)
      .value("ZERO_COPY", PjRtClient::HostBufferSemantics::kZeroCopy);

  jax::BuildWeakrefLRUCacheAPI(m);

  py::class_<PyClient, std::shared_ptr<PyClient>> py_local_client(m, "Client");
  py_local_client.def_property_readonly("platform", &PyClient::platform_name)
      .def_property_readonly("platform_version", &PyClient::platform_version)
      .def_property_readonly("runtime_type", &PyClient::runtime_type)
      .def("device_count", &PyClient::device_count)
      .def("local_device_count", &PyClient::addressable_device_count)
      .def("devices", &PyClient::Devices)
      .def("local_devices", &PyClient::LocalDevices)
      .def("live_buffers", &PyClient::LiveBuffers)
      .def("live_executables", &PyClient::LiveExecutables)
      .def("live_arrays", &PyClient::LiveArrays)
      .def("process_index", &PyClient::process_index)
      .def("host_id", &PyClient::process_index)
      .def("task_id", &PyClient::process_index)
      .def("get_default_device_assignment",
           xla::ValueOrThrowWrapper(&PyClient::GetDefaultDeviceAssignment))
      // TODO(skye): delete after all callers can handle 2D output
      .def("get_default_device_assignment",
           xla::ValueOrThrowWrapper(&PyClient::GetDefaultDeviceAssignment1D))
      .def("create_channel_handle",
           xla::ValueOrThrowWrapper(&PyClient::CreateChannelHandle))
      .def("create_device_to_host_channel_handle",
           xla::ValueOrThrowWrapper(&PyClient::CreateDeviceToHostChannelHandle))
      .def("create_host_to_device_channel_handle",
           xla::ValueOrThrowWrapper(&PyClient::CreateHostToDeviceChannelHandle))
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
      .def("compile", xla::ValueOrThrowWrapper(&PyClient::Compile),
           py::arg("computation"),
           py::arg("compile_options") = CompileOptions(),
           py::arg("host_callbacks") = std::vector<py::capsule>())
      .def("serialize_executable",
           xla::ValueOrThrowWrapper(&PyClient::SerializeExecutable))
      .def("deserialize_executable",
           xla::ValueOrThrowWrapper(
               py::overload_cast<const std::string&, CompileOptions,
                                 std::vector<py::capsule>>(
                   &PyClient::DeserializeExecutable)),
           py::arg("serialized"), py::arg("compile_options"),
           py::arg("host_callbacks") = std::vector<py::capsule>())
      // TODO(skyewm): remove when jax stop providing hlo_module
      .def("deserialize_executable",
           xla::ValueOrThrowWrapper(
               py::overload_cast<const std::string&, std::shared_ptr<HloModule>,
                                 CompileOptions, std::vector<py::capsule>>(
                   &PyClient::DeserializeExecutable)),
           py::arg("serialized"), py::arg("hlo_module"),
           py::arg("compile_options"),
           py::arg("host_callbacks") = std::vector<py::capsule>())
      .def("heap_profile", xla::ValueOrThrowWrapper(&PyClient::HeapProfile))
      // TODO(zhangqiaorjc): Experimental.
      .def("defragment",
           [](PyClient& self) { xla::ThrowIfError(self.Defragment()); })
      .def("get_emit_python_callback_descriptor",
           xla::ValueOrThrowWrapper(&PyClient::GetEmitPythonCallbackDescriptor),
           py::arg("callable"), py::arg("operand_shapes"),
           py::arg("result_shapes") = std::nullopt)
      .def("make_python_callback_from_host_send_and_recv",
           xla::ValueOrThrowWrapper(
               &PyClient::MakePythonCallbackUsingHostSendAndRecv),
           py::arg("callable"), py::arg("operand_shapes"),
           py::arg("result_shapes"), py::arg("send_channel_ids"),
           py::arg("recv_channel_ids"))
      // Deprecated: please use `get_emit_python_callback_descriptor` instead.
      .def("emit_python_callback",
           xla::ValueOrThrowWrapper(&PyClient::EmitPythonCallback),
           py::arg("callable"), py::arg("builder"), py::arg("operands"),
           py::arg("result_shapes"), py::arg("operand_layouts") = std::nullopt,
           py::arg("has_side_effects") = false);

  m.def(
      "get_tfrt_cpu_client",
      [](bool asynchronous) -> std::shared_ptr<PyClient> {
        py::gil_scoped_release gil_release;
        std::unique_ptr<PjRtClient> client =
            xla::ValueOrThrow(GetTfrtCpuClient(asynchronous));
        return std::make_shared<PyClient>(
            ifrt::PjRtClient::Create(std::move(client)));
      },
      py::arg("asynchronous") = true);
  m.def("get_interpreter_client", []() -> std::shared_ptr<PyClient> {
    py::gil_scoped_release gil_release;
    std::unique_ptr<PjRtClient> client =
        xla::ValueOrThrow(GetInterpreterClient());
    return std::make_shared<PyClient>(
        ifrt::PjRtClient::Create(std::move(client)));
  });
  m.def("load_pjrt_plugin",
        [](std::string platform_name, std::string library_path) {
          xla::ThrowIfError(pjrt::LoadPjrtPlugin(platform_name, library_path));
        });

#ifdef XLA_PYTHON_ENABLE_GPU
  py::class_<GpuAllocatorConfig> alloc_config(m, "GpuAllocatorConfig");
  alloc_config.def(py::init<>())
      .def_readwrite("kind", &GpuAllocatorConfig::kind)
      .def_readwrite("memory_fraction", &GpuAllocatorConfig::memory_fraction)
      .def_readwrite("preallocate", &GpuAllocatorConfig::preallocate);
  py::enum_<GpuAllocatorConfig::Kind>(alloc_config, "Kind")
      .value("DEFAULT", GpuAllocatorConfig::Kind::kDefault)
      .value("PLATFORM", GpuAllocatorConfig::Kind::kPlatform)
      .value("BFC", GpuAllocatorConfig::Kind::kBFC)
      .value("CUDA_ASYNC", GpuAllocatorConfig::Kind::kCudaAsync);

  m.def(
      "get_gpu_client",
      [](bool asynchronous, const GpuAllocatorConfig& allocator_config,
         std::shared_ptr<DistributedRuntimeClient> distributed_client,
         int node_id, std::optional<std::set<int>> allowed_devices,
         std::optional<std::string> platform_name)
          -> std::shared_ptr<PyClient> {
        py::gil_scoped_release gil_release;
        std::unique_ptr<PjRtClient> client =
            xla::ValueOrThrow(GetStreamExecutorGpuClient(
                asynchronous, allocator_config, std::move(distributed_client),
                node_id, allowed_devices, platform_name));
        return std::make_shared<PyClient>(
            ifrt::PjRtClient::Create(std::move(client)));
      },
      py::arg("asynchronous") = true,
      py::arg("allocator_config") = GpuAllocatorConfig(),
      py::arg("distributed_client") = nullptr, py::arg("node_id") = 0,
      py::arg("allowed_devices") = std::nullopt,
      py::arg("platform_name") = std::nullopt);
#endif  // XLA_PYTHON_ENABLE_GPU

#ifdef XLA_PYTHON_ENABLE_TPU
  m.def(
      "get_tpu_client",
      [](int max_inflight_computations) -> std::shared_ptr<PyClient> {
        py::gil_scoped_release gil_release;
        std::shared_ptr<PjRtClient> client =
            xla::ValueOrThrow(GetTpuClient(max_inflight_computations));
        return std::make_shared<PyClient>(
            ifrt::PjRtClient::Create(std::move(client)));
      },
      py::arg("max_inflight_computations") = 32);
  // TODO(b/262050449): move out from `#ifdef XLA_PYTHON_ENABLE_TPU` when
  // GetCApiClient does not depend on TPU.
  m.def(
      "get_c_api_client",
      [](std::string platform_name,
         const absl::flat_hash_map<std::string, PjRtValueType>& options)
          -> std::shared_ptr<PyClient> {
        py::gil_scoped_release gil_release;
        std::unique_ptr<PjRtClient> c_api_client =
            xla::ValueOrThrow(GetCApiClient(platform_name, options));
        return std::make_shared<PyClient>(
            ifrt::PjRtClient::Create(std::move(c_api_client)));
      },
      py::arg("platform_name"),
      py::arg("options") = absl::flat_hash_map<std::string, PjRtValueType>());
  // TODO(b/262050449): move out from `#ifdef XLA_PYTHON_ENABLE_TPU` when
  // GetCApiTopology does not depend on TPU.
  m.def("get_default_c_api_topology",
        [](std::string platform_name) -> std::unique_ptr<PjRtDeviceTopology> {
          return xla::ValueOrThrow(GetCApiTopology(platform_name));
        });
#endif  // XLA_PYTHON_ENABLE_TPU

#ifdef XLA_PYTHON_ENABLE_PLUGIN_DEVICE
  m.def("get_plugin_device_client", []() -> std::shared_ptr<PyClient> {
    py::gil_scoped_release gil_release;
    std::unique_ptr<PjRtClient> client =
        xla::ValueOrThrow(GetTfrtPluginDeviceClient());
    return std::make_shared<PyClient>(
        ifrt::PjRtClient::Create(std::move(client)));
  });
#endif  // XLA_PYTHON_ENABLE_PLUGIN_DEVICE

  TF_CHECK_OK(PyArray::RegisterTypes(m));
  jax::RegisterSharding(m);

  py::class_<CompiledMemoryStats>(m, "CompiledMemoryStats")
      .def_readwrite("generated_code_size_in_bytes",
                     &CompiledMemoryStats::generated_code_size_in_bytes)
      .def_readwrite("argument_size_in_bytes",
                     &CompiledMemoryStats::argument_size_in_bytes)
      .def_readwrite("output_size_in_bytes",
                     &CompiledMemoryStats::output_size_in_bytes)
      .def_readwrite("alias_size_in_bytes",
                     &CompiledMemoryStats::alias_size_in_bytes)
      .def_readwrite("temp_size_in_bytes",
                     &CompiledMemoryStats::temp_size_in_bytes)
      .def_property_readonly("serialized_hlo_proto",
                             [](const CompiledMemoryStats& cms) -> py::bytes {
                               return py::bytes(cms.serialized_hlo_proto);
                             })
      .def("__str__", &CompiledMemoryStats::DebugString);

  py::class_<PyExecuteResults>(m, "ExecuteResults")
      .def("__len__", [](PyExecuteResults& results) { return results.Size(); })
      .def("disassemble_into_single_device_arrays",
           [](PyExecuteResults& results) {
             return results.DisassembleIntoSingleDeviceArrays();
           })
      .def("disassemble_prefix_into_single_device_arrays",
           [](PyExecuteResults& results, size_t n) {
             return results.DisassemblePrefixIntoSingleDeviceArrays(n);
           })
      .def("consume_with_handlers",
           [](PyExecuteResults& results,
              std::vector<std::variant<const PyArrayResultHandler*, py::object>>
                  out_handlers) {
             return results.ConsumeWithHandlers(std::move(out_handlers));
           })
      .def("consume_token",
           [](PyExecuteResults& results) { return results.ConsumeToken(); });

  py::class_<PyLoadedExecutable, std::shared_ptr<PyLoadedExecutable>>
      loaded_executable(m, "LoadedExecutable");
  loaded_executable.def_property_readonly("client", &PyLoadedExecutable::client)
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
      .def("local_devices", &PyLoadedExecutable::AddressableDevices)
      .def("size_of_generated_code_in_bytes",
           &PyLoadedExecutable::SizeOfGeneratedCodeInBytes)
      .def(
          "get_compiled_memory_stats",
          xla::ValueOrThrowWrapper(&PyLoadedExecutable::GetCompiledMemoryStats))
      .def("delete", &PyLoadedExecutable::Delete)
      .def("execute_sharded_on_local_devices",
           xla::ValueOrThrowWrapper(
               &PyLoadedExecutable::ExecuteShardedOnLocalDevices),
           py::arg("arguments"))
      .def("execute_sharded_on_local_devices_with_tokens",
           xla::ValueOrThrowWrapper(
               &PyLoadedExecutable::ExecuteShardedOnLocalDevicesWithTokens),
           py::arg("arguments"))
      // TODO(parkers): Switch execute_sharded_on_local_devices* to this.
      .def("execute_sharded",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::ExecuteSharded),
           py::arg("arguments"), py::arg("with_tokens") = false)
      .def("hlo_modules",
           xla::ValueOrThrowWrapper(&PyLoadedExecutable::HloModules))
      .def("get_output_shardings", &PyLoadedExecutable::GetOutputShardings)
      .def("get_parameter_shardings",
           &PyLoadedExecutable::GetParameterShardings)
      .def("keep_alive", &PyLoadedExecutable::KeepAlive)
      .def("compile_options",
           [](const PyLoadedExecutable& self) {
             return ValueOrThrow(self.pjrt_executable()->GetCompileOptions());
           })
      .def_property_readonly("traceback", &PyLoadedExecutable::traceback)
      .def_property_readonly("fingerprint",
                             [](PyLoadedExecutable* exec) -> py::object {
                               if (exec->fingerprint().has_value()) {
                                 return py::bytes(*exec->fingerprint());
                               } else {
                                 return py::none();
                               }
                             });
  py::class_<PyToken> token(m, "Token");
  token.def("block_until_ready",
            [](PyToken& self) { xla::ThrowIfError(self.Await()); });
  py::class_<PyShardedToken> sharded_token(m, "ShardedToken");
  sharded_token.def("block_until_ready", [](PyShardedToken& self) {
    xla::ThrowIfError(self.Await());
  });
  sharded_token.def("get_token", &PyShardedToken::GetPyToken);

  m.def("buffer_to_dlpack_managed_tensor",
        xla::ValueOrThrowWrapper(BufferToDLPackManagedTensor),
        py::arg("buffer"), py::arg("take_ownership") = true);
  m.def(
      "dlpack_managed_tensor_to_buffer",
      [](const pybind11::capsule& tensor, std::shared_ptr<PyClient> cpu_client,
         std::shared_ptr<PyClient> gpu_client) {
        return xla::ValueOrThrow(DLPackManagedTensorToBuffer(
            tensor, std::move(cpu_client), std::move(gpu_client)));
      },
      py::arg("dlpack"), py::arg("cpu_backend") = nullptr,
      py::arg("gpu_backend") = nullptr);

  BuildProfilerSubmodule(&m);
  BuildOpsSubmodule(&m);
  BuildOutfeedReceiverSubmodule(&m);
  BuildPytreeSubmodule(m);
  jax::BuildJaxjitSubmodule(m);
  jax::BuildPmapSubmodule(m);
  jax::BuildPjitSubmodule(m);
  jax::BuildTransferGuardSubmodule(m);
  BuildTracebackSubmodule(m);
  BuildMlirSubmodule(m);
  BuildCustomCallShardingPybindAPI(m);

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
      [](std::string address, int num_nodes, bool use_coordination_service,
         std::optional<int> heartbeat_interval,
         std::optional<int> max_missing_heartbeats,
         std::optional<int> enumerate_devices_timeout,
         std::optional<int> shutdown_timeout)
          -> std::unique_ptr<DistributedRuntimeService> {
        DistributedRuntimeServiceImpl::Options options;
        options.num_nodes = num_nodes;
        if (heartbeat_interval.has_value()) {
          options.heartbeat_interval = absl::Seconds(*heartbeat_interval);
        }
        if (max_missing_heartbeats.has_value()) {
          options.max_missing_heartbeats = *max_missing_heartbeats;
        }
        if (enumerate_devices_timeout.has_value()) {
          options.enumerate_devices_timeout =
              absl::Seconds(*enumerate_devices_timeout);
        }
        if (shutdown_timeout.has_value()) {
          options.shutdown_timeout = absl::Seconds(*shutdown_timeout);
        }
        std::unique_ptr<DistributedRuntimeService> service =
            xla::ValueOrThrow(GetDistributedRuntimeService(
                address, options, use_coordination_service));
        return service;
      },
      py::arg("address"), py::arg("num_nodes"),
      py::arg("use_coordination_service"), py::kw_only(),
      py::arg("heartbeat_interval") = std::nullopt,
      py::arg("max_missing_heartbeats") = std::nullopt,
      py::arg("enumerate_devices_timeout") = std::nullopt,
      py::arg("shutdown_timeout") = std::nullopt);

  m.def(
      "get_distributed_runtime_client",
      [](std::string address, int node_id, bool use_coordination_service,
         std::optional<int> rpc_timeout, std::optional<int> init_timeout,
         std::optional<int> shutdown_timeout,
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
        return GetDistributedRuntimeClient(address, options,
                                           use_coordination_service);
      },
      py::arg("address"), py::arg("node_id"),
      py::arg("use_coordination_service"), py::kw_only(),
      py::arg("rpc_timeout") = std::nullopt,
      py::arg("init_timeout") = std::nullopt,
      py::arg("shutdown_timeout") = std::nullopt,
      py::arg("heartbeat_interval") = std::nullopt,
      py::arg("max_missing_heartbeats") = std::nullopt,
      py::arg("missed_heartbeat_callback") = std::nullopt,
      py::arg("shutdown_on_destruction") = std::nullopt);

  m.def("collect_garbage", []() { GlobalPyRefManager()->CollectGarbage(); });

  m.def("is_optimized_build", &IsOptimizedBuild);

  m.def("json_to_pprof_profile", xla::ValueOrThrowWrapper(JsonToPprofProfile),
        "Encodes the JSON representation of a pprof Profile into its binary "
        "protocol buffer encoding.");
  m.def("pprof_profile_to_json", xla::ValueOrThrowWrapper(PprofProfileToJson),
        "Decodes an uncompressed pprof Profile protocol buffer into a JSON "
        "representation");

  py::class_<PjRtDeviceTopology>(m, "DeviceTopology")
      .def_property_readonly(
          "platform",
          [](PjRtDeviceTopology& topology) { return topology.platform_name(); })
      .def_property_readonly("platform_version",
                             [](PjRtDeviceTopology& topology) {
                               return topology.platform_version();
                             })
      .def_property_readonly("device_attributes",
                             [](PjRtDeviceTopology& topology) {
                               return py::cast(topology.DeviceAttributes());
                             });

  py::class_<PjRtExecutable, std::shared_ptr<PjRtExecutable>>(m, "Executable")
      .def("hlo_modules", &PjRtExecutable::GetHloModules)
      .def("get_output_shardings", &PjRtExecutable::GetOutputShardings)
      .def("get_parameter_shardings", &PjRtExecutable::GetParameterShardings)
      .def("get_compiled_memory_stats", &PjRtExecutable::GetCompiledMemoryStats)
      .def("compile_options", &PjRtExecutable::GetCompileOptions)
      .def("serialize", [](const PjRtExecutable& exec) -> py::bytes {
        return ValueOrThrow(exec.SerializeExecutable());
      });

  m.def(
      "compile",
      [](const PjRtDeviceTopology& topology, std::string mlir_module,
         CompileOptions options) -> std::shared_ptr<PjRtExecutable> {
        std::unique_ptr<PjRtExecutable> executable;
        std::optional<std::string> fingerprint;
        {
          py::gil_scoped_release gil_release;
          mlir::MLIRContext context;
          mlir::OwningOpRef<mlir::ModuleOp> module =
              xla::ValueOrThrow(ParseMlirModuleString(mlir_module, context));
          executable = xla::ValueOrThrow(
              PjRtCompile(std::move(options), module.get(), topology));
        }
        return std::shared_ptr<PjRtExecutable>(std::move(executable));
      },
      py::arg("topology"), py::arg("computation"),
      py::arg("compile_options") = CompileOptions());

  m.def("is_asan", IsAsan);
  m.def("is_msan", IsMsan);
  m.def("is_tsan", IsTsan);
  m.def("is_sanitized", IsSanitized);

  m.attr("batched_device_put") = py::cpp_function(
      [](py::object aval, py::object sharding, std::vector<py::object> xs,
         std::vector<ClientAndPtr<PjRtDevice>> dst_devices, bool committed,
         bool force_copy,
         PjRtClient::HostBufferSemantics host_buffer_semantics) -> PyArray {
        return ValueOrThrow(PyArray::BatchedDevicePut(
            std::move(aval), std::move(sharding), std::move(xs),
            std::move(dst_devices), committed, force_copy,
            host_buffer_semantics, jax::GetEnableX64()));
      },
      py::arg("aval"), py::arg("sharding"), py::arg("xs"), py::arg("devices"),
      py::arg("committed") = true, py::arg("force_copy") = false,
      py::arg("host_buffer_semantics") =
          PjRtClient::HostBufferSemantics::kZeroCopy);
}  // NOLINT(readability/fn_size)

}  // namespace xla
