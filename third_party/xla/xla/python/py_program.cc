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

#include "xla/python/py_program.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/custom_call_program.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/plugin_program.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/python/py_device.h"
#include "xla/python/py_device_list.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/sharding.h"
#include "xla/python/types.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace nb = ::nanobind;

namespace {

// Gets `ifrt::DeviceList` from a sequence of JAX devices.
absl::StatusOr<tsl::RCReference<ifrt::DeviceList>> GetDeviceList(
    nb::sequence devices) {
  tsl::RCReference<ifrt::DeviceList> ifrt_device_list;
  if (devices.type().is(jax::PyDeviceList::type())) {
    return nb::cast<const jax::PyDeviceList*>(devices)->ifrt_device_list();
  } else {
    auto py_devices = nb::cast<std::vector<nb_class_ptr<PyDevice>>>(devices);
    ifrt::BasicDeviceList::Devices ifrt_devices;
    ifrt_devices.reserve(py_devices.size());
    for (const nb_class_ptr<PyDevice>& py_device : py_devices) {
      ifrt_devices.push_back(py_device->device());
    }
    return ifrt::BasicDeviceList::Create(std::move(ifrt_devices));
  }
}

// Gets `xla::HloSharding` from a JAX Sharding.
xla::HloSharding GetXlaHloSharding(nb::handle sharding,
                                   int64_t num_dimensions) {
  if (sharding.type().is(jax::GSPMDSharding::type())) {
    return nb::cast<jax::GSPMDSharding*>(sharding)->hlo_sharding();
  } else {
    return nb::cast<xla::HloSharding>(
        sharding.attr("_to_xla_hlo_sharding")(num_dimensions));
  }
}

// Gets `ifrt::DeviceList` from a JAX Sharding.
absl::StatusOr<tsl::RCReference<ifrt::DeviceList>> GetIfrtDeviceList(
    nb::handle sharding) {
  if (sharding.type().is(jax::NamedSharding::type())) {
    TF_ASSIGN_OR_RETURN(
        auto ns_device_list,
        nb::cast<const jax::NamedSharding*>(sharding)->internal_device_list());
    return ns_device_list->ifrt_device_list();
  } else if (sharding.type().is(jax::SingleDeviceSharding::type())) {
    return nb::cast<const jax::SingleDeviceSharding*>(sharding)
        ->internal_device_list()
        ->ifrt_device_list();
  } else if (sharding.type().is(jax::PmapSharding::type())) {
    return nb::cast<const jax::PmapSharding*>(sharding)
        ->internal_device_list()
        ->ifrt_device_list();
  } else if (sharding.type().is(jax::GSPMDSharding::type())) {
    return nb::cast<const jax::GSPMDSharding*>(sharding)
        ->internal_device_list()
        ->ifrt_device_list();
  } else {
    return nb::cast<const jax::PyDeviceList*>(
               sharding.attr("_internal_device_list"))
        ->ifrt_device_list();
  }
}

// Gets `ifrt::MemoryKind` from a JAX Sharding.
ifrt::MemoryKind GetIfrtMemoryKind(nb::handle sharding) {
  auto memory_kind = sharding.attr("memory_kind");
  if (memory_kind.is_none()) {
    return ifrt::MemoryKind();
  } else {
    return ifrt::MemoryKind(nb::cast<std::string>(memory_kind));
  }
}

// Makes `ifrt::Sharding` from a JAX Sharding. It requires the number of shape
// dimensions, which may become necessary when building an HLO sharding.
absl::StatusOr<std::shared_ptr<const ifrt::Sharding>> GetIfrtSharding(
    nb::handle sharding, int64_t num_dimensions) {
  auto ifrt_memory_kind = GetIfrtMemoryKind(sharding);
  std::shared_ptr<const ifrt::Sharding> ifrt_sharding;
  if (sharding.type().is(jax::SingleDeviceSharding::type())) {
    TF_ASSIGN_OR_RETURN(auto ifrt_device_list,
                        nb::cast<const jax::SingleDeviceSharding*>(sharding)
                            ->internal_device_list()
                            ->ifrt_device_list());
    return ifrt::SingleDeviceSharding::Create(
        ifrt_device_list->devices().front(), ifrt_memory_kind);
  } else {
    TF_ASSIGN_OR_RETURN(auto ifrt_device_list, GetIfrtDeviceList(sharding));
    auto xla_hlo_sharding = GetXlaHloSharding(sharding, num_dimensions);
    return ifrt::HloSharding::Create(std::move(ifrt_device_list),
                                     ifrt_memory_kind,
                                     std::move(xla_hlo_sharding));
  }
}

// Gets `ifrt::ArraySpec`s from a sequence of JAX avals (e.g.,
// `jax.ShapeDtypeStruct`).
absl::StatusOr<std::vector<ifrt::ArraySpec>> GetIfrtArraySpecs(
    nb::sequence avals) {
  std::vector<ifrt::ArraySpec> ifrt_array_specs;
  ifrt_array_specs.reserve(nb::len(avals));
  for (nb::handle aval : avals) {
    ifrt::Shape ifrt_shape(nb::cast<std::vector<int64_t>>(aval.attr("shape")));
    TF_ASSIGN_OR_RETURN(
        auto ifrt_dtype,
        DtypeToIfRtDType(nb::cast<nb_dtype>(aval.attr("dtype"))));
    TF_ASSIGN_OR_RETURN(
        auto ifrt_sharding,
        GetIfrtSharding(aval.attr("sharding"), ifrt_shape.dims().size()));
    ifrt_array_specs.push_back(ifrt::ArraySpec{
        ifrt_dtype, std::move(ifrt_shape), std::move(ifrt_sharding)});
  }
  return ifrt_array_specs;
}

absl::StatusOr<std::unique_ptr<xla::ifrt::Program>> MakePluginProgramFromString(
    std::string data) {
  auto plugin_program = std::make_unique<xla::ifrt::PluginProgram>();
  plugin_program->data = std::move(data);
  return plugin_program;
}

absl::StatusOr<std::unique_ptr<xla::ifrt::Program>> MakePluginProgramFromBytes(
    nb::bytes data) {
  auto plugin_program = std::make_unique<xla::ifrt::PluginProgram>();
  plugin_program->data = std::string(data.c_str(), data.size());
  return plugin_program;
}

absl::StatusOr<std::unique_ptr<ifrt::CompileOptions>>
MakeColocatedPythonCompileOptions() {
  return std::make_unique<ifrt::CustomCallCompileOptions>();
}

absl::StatusOr<std::unique_ptr<ifrt::CompileOptions>>
MakePluginCompileOptions() {
  return std::make_unique<ifrt::PluginCompileOptions>();
}

absl::StatusOr<std::unique_ptr<ifrt::Program>> MakeHloProgram(
    absl::string_view mlir_module) {
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(mlir_module, *context));
  return std::make_unique<xla::ifrt::HloProgram>(std::move(context),
                                                 std::move(module));
}

absl::StatusOr<std::unique_ptr<ifrt::Program>> MakeHloProgramFromString(
    std::string mlir_module) {
  return MakeHloProgram(mlir_module);
}

absl::StatusOr<std::unique_ptr<ifrt::Program>> MakeHloProgramFromBytes(
    nb::bytes mlir_module) {
  return MakeHloProgram(
      absl::string_view(mlir_module.c_str(), mlir_module.size()));
}

absl::StatusOr<std::unique_ptr<ifrt::CompileOptions>> MakeXlaCompileOptions(
    CompileOptions options, std::vector<nb::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that were
  // created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()` or
  // `PyClient::GetEmitPythonCallbackDescriptor()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(tsl::FormRef(
        static_cast<ifrt::LoadedHostCallback*>(host_callback.data())));
  }
  return std::make_unique<ifrt::XlaCompileOptions>(
      std::move(options), std::move(ifrt_loaded_host_callbacks));
}

constexpr absl::string_view kColocatedPythonProgramType =
    "jax_colocated_python_v0.0.1";

absl::StatusOr<std::unique_ptr<ifrt::Program>> MakeColocatedPythonProgram(
    std::string name, nb::bytes picked_function, nb::sequence devices,
    nb::sequence input_avals, nb::sequence output_avals) {
  auto ifrt_serialized_program_text = absl::MakeCordFromExternal(
      absl::string_view(reinterpret_cast<const char*>(picked_function.data()),
                        picked_function.size()),
      /*releaser=*/[picked_function](absl::string_view) mutable {
        GlobalPyRefManager()->AddGarbage(std::move(picked_function));
      });
  TF_ASSIGN_OR_RETURN(auto ifrt_device_list, GetDeviceList(devices));
  TF_ASSIGN_OR_RETURN(auto ifrt_input_specs, GetIfrtArraySpecs(input_avals));
  TF_ASSIGN_OR_RETURN(auto ifrt_output_specs, GetIfrtArraySpecs(output_avals));
  return std::make_unique<ifrt::CustomCallProgram>(
      std::string(kColocatedPythonProgramType), std::move(name),
      std::move(ifrt_serialized_program_text), std::move(ifrt_device_list),
      std::move(ifrt_input_specs), std::move(ifrt_output_specs));
}

}  // namespace

void BuildIfrtProgramsSubmodule(nanobind::module_& m) {
  auto sub_module = m.def_submodule("ifrt_programs");
  nb::class_<ifrt::Program> ifrt_program_base_class(sub_module, "Program");
  nb::class_<ifrt::CompileOptions> ifrt_compile_options_base_class(
      sub_module, "CompileOptions");
  sub_module
      .def("make_hlo_program", ValueOrThrowWrapper(MakeHloProgramFromString),
           nb::arg("mlir_module"))
      .def("make_hlo_program", ValueOrThrowWrapper(MakeHloProgramFromBytes),
           nb::arg("mlir_module"))
      .def("make_colocated_python_program",
           ValueOrThrowWrapper(MakeColocatedPythonProgram), nb::arg("name"),
           nb::arg("pickled_function"), nb::arg("devices"),
           nb::arg("input_avals"), nb::arg("output_avals"))
      .def("make_plugin_program",
           ValueOrThrowWrapper(MakePluginProgramFromString), nb::arg("data"))
      .def("make_plugin_program",
           ValueOrThrowWrapper(MakePluginProgramFromBytes), nb::arg("data"))
      .def("make_xla_compile_options",
           ValueOrThrowWrapper(MakeXlaCompileOptions), nb::arg("options"),
           nb::arg("host_callbacks"))
      .def("make_colocated_python_compile_options",
           ValueOrThrowWrapper(MakeColocatedPythonCompileOptions))
      .def("make_plugin_compile_options",
           ValueOrThrowWrapper(MakePluginCompileOptions));
}

}  // namespace xla
