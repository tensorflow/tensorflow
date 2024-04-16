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

#include "xla/python/py_compile_only_client.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/literal.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/py_client.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {

namespace {

class PjRtCompileOnlyDevice : public PjRtDevice {
 public:
  explicit PjRtCompileOnlyDevice(const PjRtDeviceDescription* description)
      : description_(std::move(description)) {}

  const PjRtDeviceDescription& description() const override {
    return *description_;
  }

  PjRtClient* client() const override { return nullptr; }
  bool IsAddressable() const override { return false; }
  int local_hardware_id() const override {
    return local_hardware_id_typed().value();
  }

  PjRtLocalDeviceId local_device_id() const override {
    return PjRtLocalDeviceId(local_hardware_id_typed().value());
  }

  PjRtLocalHardwareId local_hardware_id_typed() const override {
    return PjRtLocalHardwareId(-1);
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }
  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented("TransferToInfeed is not supported");
  }
  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return Unimplemented("TransferFromOutfeed is not supported");
  }
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return {};
  }
  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override {
    return Unimplemented("default_memory_space is not supported");
  }

 private:
  const PjRtDeviceDescription* description_;
};

class InvalidIfrtCompiler final
    : public llvm::RTTIExtends<InvalidIfrtCompiler, ifrt::Compiler> {
 public:
  absl::StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> Compile(
      std::unique_ptr<ifrt::Program> program,
      std::unique_ptr<ifrt::CompileOptions> options) override {
    return Unimplemented("Compile not implemented.");
  }

  absl::StatusOr<std::unique_ptr<ifrt::LoadedExecutable>>
  DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<ifrt::DeserializeExecutableOptions> options) override {
    return Unimplemented("DeserializeLoadedExecutable not implemented.");
  }

  static char ID;  // NOLINT
};
char InvalidIfrtCompiler::ID = 0;  // NOLINT

class CompileOnlyIfRtClient final
    : public llvm::RTTIExtends<CompileOnlyIfRtClient, ifrt::Client> {
 public:
  explicit CompileOnlyIfRtClient(
      std::shared_ptr<PjRtTopologyDescription> topology)
      : topology_(std::move(topology)),
        descriptions_(topology_->DeviceDescriptions()) {
    for (auto& description : descriptions_) {
      owned_devices_.push_back(
          std::make_unique<PjRtCompileOnlyDevice>(description.get()));
      devices_.push_back(owned_devices_.back().get());
    }
  }

  absl::StatusOr<tsl::RCReference<ifrt::Array>> MakeArrayFromHostBuffer(
      const void* data, ifrt::DType dtype, ifrt::Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      std::shared_ptr<const ifrt::Sharding> sharding,
      HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) override {
    return Unimplemented(
        "MakeArrayFromHostBuffer not available with compile-only client.");
  }

  absl::StatusOr<tsl::RCReference<ifrt::Array>>
  AssembleArrayFromSingleDeviceArrays(
      ifrt::Shape shape, std::shared_ptr<const ifrt::Sharding> sharding,
      absl::Span<tsl::RCReference<ifrt::Array>> arrays,
      ifrt::ArrayCopySemantics semantics) override {
    return Unimplemented(
        "AssembleArrayFromSingleDeviceArrays not available with compile-only "
        "client.");
  }

  absl::StatusOr<tsl::RCReference<ifrt::Tuple>> MakeTuple(
      absl::Span<tsl::RCReference<ifrt::Value>> values) override {
    return Unimplemented("MakeTuple not available with compile-only client.");
  }

  absl::string_view runtime_type() const override {
    return "compile_only_runtime";
  }

  absl::string_view platform_name() const override {
    return topology_->platform_name();
  }
  absl::string_view platform_version() const override {
    return topology_->platform_version();
  }
  ifrt::PlatformId platform_id() const override {
    return topology_->platform_id();
  }
  absl::flat_hash_map<std::string, ClientAttribute> attributes()
      const override {
    return {};
  }

  int device_count() const override { return devices().size(); }
  int addressable_device_count() const override { return 0; }
  absl::Span<ifrt::Device* const> devices() const override { return devices_; }
  absl::Span<ifrt::Device* const> addressable_devices() const override {
    return {};
  }
  int process_index() const override { return 0; }
  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return Unimplemented(
        "GetDefaultDeviceAssignment not available with compile-only client.");
  }
  absl::StatusOr<ifrt::Device*> LookupDevice(int device_id) const override {
    return Unimplemented(
        "LookupDevice not available with compile-only client.");
  }

  absl::StatusOr<ifrt::Device*> LookupAddressableDevice(
      int local_hardware_id) const override {
    return Unimplemented(
        "LookupAddressableDevice not available with compile-only client.");
  }

  ifrt::Compiler* GetDefaultCompiler() override { return &default_compiler_; }

  static char ID;  // NOLINT

  const PjRtTopologyDescription& topology() const { return *topology_; }

  absl::StatusOr<std::shared_ptr<const xla::PjRtTopologyDescription>>
  GetTopologyForDevices(const xla::ifrt::DeviceList& devices) const override {
    return topology_;
  }

  absl::StatusOr<std::unique_ptr<PjRtLayout>> GetDefaultLayoutForDevice(
      ifrt::DType dtype, absl::Span<const int64_t> dims,
      ifrt::Device* device) const override {
    return absl::UnimplementedError(
        "GetDefaultLayout not supported for CompileOnlyIfRtClient.");
  }

 private:
  InvalidIfrtCompiler default_compiler_;
  std::shared_ptr<PjRtTopologyDescription> topology_;
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> descriptions_;
  std::vector<std::unique_ptr<PjRtCompileOnlyDevice>> owned_devices_;
  std::vector<PjRtDevice*> devices_;
};

char CompileOnlyIfRtClient::ID = 0;  // NOLINT

class CompileOnlyPyClient : public PyClient {
 public:
  using PyClient::PyClient;

  static nb_class_ptr<PyClient> Make(
      std::shared_ptr<PjRtTopologyDescription> topology) {
    auto client =
        nb::borrow<nb_class_ptr<PyClient>>(make_nb_class<CompileOnlyPyClient>(
            std::make_unique<CompileOnlyIfRtClient>(std::move(topology))));
    CompileOnlyPyClient::Initialize(client);
    return client;
  }

  absl::StatusOr<std::shared_ptr<PjRtExecutable>> CompileUnloaded(
      std::string_view mlir_module, CompileOptions options,
      std::vector<nb::capsule> host_callbacks) {
    if (!host_callbacks.empty()) {
      return Unimplemented(
          "Compiling with host_callbacks not available with compile-only "
          "client.");
    }
    nb::gil_scoped_release gil_release;
    mlir::MLIRContext context;
    TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                        ParseMlirModuleString(mlir_module, context));
    auto* ifrt_client =
        llvm::dyn_cast_or_null<CompileOnlyIfRtClient>(this->ifrt_client());
    CHECK(ifrt_client) << "CompileOnlyPyClient requires ifrt_client be a "
                          "CompileOnlyIfRtClient";
    return PjRtCompile(std::move(options), module.get(),
                       ifrt_client->topology());
  }

 private:
  static void Initialize(nb_class_ptr<PyClient> client) {
    PyClient::Initialize(client);
  }
};

}  // namespace

nb_class_ptr<PyClient> MakeCompileOnlyClient(
    std::shared_ptr<PjRtTopologyDescription> topology) {
  return CompileOnlyPyClient::Make(std::move(topology));
}

void RegisterCompileOnlyClient(nb::module_& m) {
  nb::class_<CompileOnlyPyClient, PyClient>(m, "CompileOnlyPyClient")
      .def(
          "compile",
          [](CompileOnlyPyClient& self, nb::bytes mlir_module,
             CompileOptions options, std::vector<nb::capsule> host_callbacks) {
            return ValueOrThrow(self.CompileUnloaded(
                std::string_view(mlir_module.c_str(), mlir_module.size()),
                std::move(options), std::move(host_callbacks)));
          },
          nb::arg("computation"), nb::arg("compile_options") = CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>())
      .def(
          "compile", ValueOrThrowWrapper(&CompileOnlyPyClient::CompileUnloaded),
          nb::arg("computation"), nb::arg("compile_options") = CompileOptions(),
          nb::arg("host_callbacks") = std::vector<nb::capsule>());
}

}  // namespace xla
