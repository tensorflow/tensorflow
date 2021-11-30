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
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl_bind.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"
#include "tensorflow/compiler/xla/pjrt/distributed/service.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/interpreter_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/compiler/xla/python/dlpack.h"
#include "tensorflow/compiler/xla/python/jax_jit.h"
#include "tensorflow/compiler/xla/python/mlir.h"
#include "tensorflow/compiler/xla/python/ops.h"
#include "tensorflow/compiler/xla/python/outfeed_receiver_py.h"
#include "tensorflow/compiler/xla/python/pmap_lib.h"
#include "tensorflow/compiler/xla/python/profiler.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/xla_compiler.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/python/lib/core/bfloat16.h"

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

}  // namespace

PYBIND11_MODULE(xla_extension, m) {
  CHECK(tensorflow::RegisterNumpyBfloat16());

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
      .value("BF16", BF16)
      .value("F32", F32)
      .value("F64", F64)
      .value("C64", C64)
      .value("C128", C128)
      .value("TUPLE", TUPLE)
      .value("OPAQUE_TYPE", OPAQUE_TYPE)
      .value("TOKEN", TOKEN);

  m.def("bfloat16_dtype",
        []() { return py::handle(tensorflow::Bfloat16Dtype()); });

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
                             [](const PjRtDevice& device) {
                               return device.client()->platform_name();
                             })
      .def_property_readonly("device_kind", &PjRtDevice::device_kind)
      .def_property_readonly(
          "client",
          [](const ClientAndPtr<PjRtDevice>& device) { return device.client; })
      .def("__str__", &PjRtDevice::DebugString)
      .def("transfer_to_infeed",
           [](PjRtDevice& device, const LiteralSlice& literal) {
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             return device.TransferToInfeed(literal);
           })
      .def("transfer_from_outfeed",
           [](PjRtDevice& device, const Shape& shape) -> StatusOr<py::object> {
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
               TF_RETURN_IF_ERROR(device.TransferFromOutfeed(literal.get()));
             }
             return LiteralToPython(std::move(literal));
           })
      .def("live_buffers", [](const ClientAndPtr<PjRtDevice>& device) {
        return device.client->LiveBuffersOnDevice(device.get());
      });

  py::class_<CpuDevice, PjRtDevice, ClientAndPtr<CpuDevice>>(m, "CpuDevice")
      .def("__repr__", [](const CpuDevice& device) {
        return absl::StrFormat("CpuDevice(id=%i)", device.id());
      });

  py::class_<GpuDevice, PjRtDevice, ClientAndPtr<GpuDevice>>(m, "GpuDevice")
      .def_property_readonly("device_vendor", &GpuDevice::device_vendor)
      .def("__repr__", [](const GpuDevice& device) {
        return absl::StrFormat("GpuDevice(id=%i, process_index=%i)",
                               device.id(), device.process_index());
      });

  py::class_<PjRtTpuDevice, PjRtDevice, ClientAndPtr<PjRtTpuDevice>>(
      m, "TpuDevice")
      .def_property_readonly(
          "coords",
          [](const PjRtTpuDevice& device) -> pybind11::tuple {
            return SpanToTuple(absl::MakeConstSpan(device.coords()));
          },
          "The coordinates of this TpuDevice's chip in the TPU mesh network.")
      .def_property_readonly(
          "core_on_chip", &PjRtTpuDevice::core_on_chip,
          "The index of this TpuDevice's core on the TPU chip.")
      .def("__repr__", [](const PjRtTpuDevice& device) {
        return absl::StrFormat(
            "TpuDevice(id=%i, process_index=%i, coords=(%s), core_on_chip=%i)",
            device.id(), device.process_index(),
            absl::StrJoin(device.coords(), ","), device.core_on_chip());
      });

  // Local XLA client methods.

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

  py::enum_<PjRtClient::HostBufferSemantics>(m, "HostBufferSemantics")
      .value("IMMUTABLE_ONLY_DURING_CALL",
             PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall)
      .value("IMMUTABLE_UNTIL_TRANSFER_COMPLETES",
             PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes)
      .value("ZERO_COPY", PjRtClient::HostBufferSemantics::kZeroCopy);

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
      .def("process_index", &PyClient::process_index)
      .def("host_id", &PyClient::process_index)
      .def("task_id", &PyClient::process_index)
      .def("get_default_device_assignment",
           &PyClient::GetDefaultDeviceAssignment)
      // TODO(skye): delete after all callers can handle 2D output
      .def("get_default_device_assignment",
           &PyClient::GetDefaultDeviceAssignment1D)
      .def("create_channel_handle", &PyClient::CreateChannelHandle)
      .def("create_device_to_host_channel_handle",
           &PyClient::CreateDeviceToHostChannelHandle)
      .def("create_host_to_device_channel_handle",
           &PyClient::CreateHostToDeviceChannelHandle)
      .def("buffer_from_pyval", &PyClient::BufferFromPyval, py::arg("argument"),
           py::arg("device") = nullptr, py::arg("force_copy") = false,
           py::arg("host_buffer_semantics") =
               PjRtClient::HostBufferSemantics::kZeroCopy)
      .def("compile", &PyClient::Compile, py::arg("computation"),
           py::arg("compile_options") = CompileOptions())
      .def("compile", &PyClient::CompileMlir, py::arg("computation"),
           py::arg("compile_options") = CompileOptions())
      .def("serialize_executable", &PyClient::SerializeExecutable)
      .def("deserialize_executable",
           py::overload_cast<const std::string&, CompileOptions>(
               &PyClient::DeserializeExecutable))
      // TODO(skyewm): remove when jax stop providing hlo_module
      .def("deserialize_executable",
           py::overload_cast<const std::string&, std::shared_ptr<HloModule>,
                             CompileOptions>(&PyClient::DeserializeExecutable))
      .def("heap_profile", &PyClient::HeapProfile)
      // TODO(zhangqiaorjc): Experimental.
      .def("defragment", &PyClient::Defragment)
      .def("emit_python_callback", &PyClient::EmitPythonCallback,
           py::arg("callable"), py::arg("builder"), py::arg("operands"),
           py::arg("result_shapes"), py::arg("operand_layouts") = absl::nullopt,
           py::arg("has_side_effects") = false);

  m.def(
      "get_cpu_client",
      [](bool asynchronous) -> StatusOr<std::shared_ptr<PyClient>> {
        py::gil_scoped_release gil_release;
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                            GetCpuClient(asynchronous));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true);
  m.def(
      "get_tfrt_cpu_client",
      [](bool asynchronous) -> StatusOr<std::shared_ptr<PyClient>> {
        py::gil_scoped_release gil_release;
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                            GetTfrtCpuClient(asynchronous));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true);
  m.def("get_interpreter_client", []() -> StatusOr<std::shared_ptr<PyClient>> {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                        GetInterpreterClient());
    return std::make_shared<PyClient>(std::move(client));
  });
  m.def(
      "get_gpu_client",
      [](bool asynchronous, const GpuAllocatorConfig& allocator_config,
         std::shared_ptr<DistributedRuntimeClient> distributed_client,
         int node_id) -> StatusOr<std::shared_ptr<PyClient>> {
        py::gil_scoped_release gil_release;
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<PjRtClient> client,
            GetGpuClient(asynchronous, allocator_config,
                         std::move(distributed_client), node_id));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true,
      py::arg("allocator_config") = GpuAllocatorConfig(),
      py::arg("distributed_client") = nullptr, py::arg("node_id") = 0);
  m.def(
      "get_tpu_client",
      [](int max_inflight_computations) -> StatusOr<std::shared_ptr<PyClient>> {
        py::gil_scoped_release gil_release;
        TF_ASSIGN_OR_RETURN(std::shared_ptr<PjRtClient> client,
                            GetTpuClient(max_inflight_computations));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("max_inflight_computations") = 32);

  TF_CHECK_OK(PyBuffer::RegisterTypes(m));

  py::class_<PyExecutable, std::shared_ptr<PyExecutable>> executable(
      m, "Executable");
  executable.def_property_readonly("client", &PyExecutable::client)
      .def("local_logical_device_ids",
           [](PyExecutable* exec) {
             auto span = exec->addressable_device_logical_ids();
             // Not on dispatch critical path, so ok to have heap allocation.
             std::vector<std::pair<int, int>> addressable_device_logic_ids;
             addressable_device_logic_ids.reserve(span.size());
             for (const auto& logical_device_id : span) {
               addressable_device_logic_ids.push_back(std::make_pair(
                   logical_device_id.replica, logical_device_id.partition));
             }
           })
      .def("local_devices", &PyExecutable::AddressableDevices)
      .def("size_of_generated_code_in_bytes",
           &PyExecutable::SizeOfGeneratedCodeInBytes)
      .def("delete", &PyExecutable::Delete)
      .def("execute", &PyExecutable::Execute, py::arg("arguments"))
      .def("execute_sharded_on_local_devices",
           &PyExecutable::ExecuteShardedOnLocalDevices, py::arg("arguments"))
      .def("hlo_modules", &PyExecutable::HloModules)
      .def("keep_alive", &PyExecutable::KeepAlive)
      .def_property_readonly("traceback", &PyExecutable::traceback)
      .def_property_readonly("fingerprint",
                             [](PyExecutable* exec) -> py::object {
                               if (exec->fingerprint().has_value()) {
                                 return py::bytes(*exec->fingerprint());
                               } else {
                                 return py::none();
                               }
                             });

  m.def("buffer_to_dlpack_managed_tensor", BufferToDLPackManagedTensor,
        py::arg("buffer"), py::arg("take_ownership") = true);
  m.def("dlpack_managed_tensor_to_buffer", DLPackManagedTensorToBuffer,
        py::arg("dlpack"), py::arg("cpu_backend") = nullptr,
        py::arg("gpu_backend") = nullptr);

  BuildProfilerSubmodule(&m);
  BuildOpsSubmodule(&m);
  BuildOutfeedReceiverSubmodule(&m);
  BuildPytreeSubmodule(m);
  jax::BuildJaxjitSubmodule(m);
  jax::BuildPmapSubmodule(m);
  BuildTracebackSubmodule(m);
  BuildMlirSubmodule(m);

  py::class_<DistributedRuntimeService,
             std::unique_ptr<DistributedRuntimeService>>
      distributed_runtime_service(m, "DistributedRuntimeService");
  distributed_runtime_service.def("shutdown",
                                  &DistributedRuntimeService::Shutdown);
  py::class_<DistributedRuntimeClient,
             std::shared_ptr<DistributedRuntimeClient>>
      distributed_runtime_client(m, "DistributedRuntimeClient");
  distributed_runtime_client.def("connect", &DistributedRuntimeClient::Connect)
      .def("shutdown", &DistributedRuntimeClient::Shutdown);

  m.def(
      "get_distributed_runtime_service",
      [](std::string address, int num_nodes,
         absl::optional<int> heartbeat_interval,
         absl::optional<int> max_missing_heartbeats,
         absl::optional<int> enumerate_devices_timeout,
         absl::optional<int> shutdown_timeout)
          -> StatusOr<std::unique_ptr<DistributedRuntimeService>> {
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
        TF_ASSIGN_OR_RETURN(std::unique_ptr<DistributedRuntimeService> service,
                            GetDistributedRuntimeService(address, options));
        return service;
      },
      py::arg("address"), py::arg("num_nodes"), py::kw_only(),
      py::arg("heartbeat_interval") = absl::nullopt,
      py::arg("max_missing_heartbeats") = absl::nullopt,
      py::arg("enumerate_devices_timeout") = absl::nullopt,
      py::arg("shutdown_timeout") = absl::nullopt);

  m.def(
      "get_distributed_runtime_client",
      [](std::string address, int node_id, absl::optional<int> rpc_timeout,
         absl::optional<int> init_timeout, absl::optional<int> shutdown_timeout,
         absl::optional<int> heartbeat_interval,
         absl::optional<int> max_missing_heartbeats,
         absl::optional<std::function<void(xla::Status,
                                           bool coordinator_reported_failure)>>
             missed_heartbeat_callback,
         absl::optional<bool> shutdown_on_destruction)
          -> StatusOr<std::shared_ptr<DistributedRuntimeClient>> {
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
      py::arg("rpc_timeout") = absl::nullopt,
      py::arg("init_timeout") = absl::nullopt,
      py::arg("shutdown_timeout") = absl::nullopt,
      py::arg("heartbeat_interval") = absl::nullopt,
      py::arg("max_missing_heartbeats") = absl::nullopt,
      py::arg("missed_heartbeat_callback") = absl::nullopt,
      py::arg("shutdown_on_destruction") = absl::nullopt);

  m.def("collect_garbage", []() { GlobalPyRefManager()->CollectGarbage(); });

  m.def("is_optimized_build", &IsOptimizedBuild);
}  // NOLINT(readability/fn_size)

}  // namespace xla
