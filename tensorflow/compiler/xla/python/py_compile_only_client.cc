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

#include "tensorflow/compiler/xla/python/py_compile_only_client.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/tsl/python/lib/core/numpy.h"  //NOLINT

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
  int local_hardware_id() const override { return -1; }
  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }
  Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented("TransferToInfeed is not supported");
  }
  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return Unimplemented("TransferFromOutfeed is not supported");
  }

 private:
  const PjRtDeviceDescription* description_;
};

class InvalidIfrtCompiler final
    : public llvm::RTTIExtends<InvalidIfrtCompiler, ifrt::Compiler> {
 public:
  StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module,
      std::unique_ptr<ifrt::CompileOptions> options) override {
    return Unimplemented("Compile not implemented.");
  }

  StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override {
    return Unimplemented("DeserializeLoadedExecutable not implemented.");
  }

  static char ID;  // NOLINT
};
char InvalidIfrtCompiler::ID = 0;

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

  StatusOr<tsl::RCReference<ifrt::Array>> MakeArrayFromHostBuffer(
      const void* data, ifrt::DType dtype, ifrt::Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      std::shared_ptr<const ifrt::Sharding> sharding,
      HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) override {
    return Unimplemented(
        "MakeArrayFromHostBuffer not available with compile-only client.");
  }

  StatusOr<tsl::RCReference<ifrt::Array>> AssembleArrayFromSingleDeviceArrays(
      ifrt::Shape shape, std::shared_ptr<const ifrt::Sharding> sharding,
      absl::Span<tsl::RCReference<ifrt::Array>> arrays,
      ifrt::ArrayCopySemantics semantics) override {
    return Unimplemented(
        "AssembleArrayFromSingleDeviceArrays not available with compile-only "
        "client.");
  }

  StatusOr<tsl::RCReference<ifrt::Tuple>> MakeTuple(
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

  int device_count() const override { return devices().size(); }
  int addressable_device_count() const override { return 0; }
  absl::Span<ifrt::Device* const> devices() const override { return devices_; }
  absl::Span<ifrt::Device* const> addressable_devices() const override {
    return {};
  }
  int process_index() const override { return 0; }
  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return Unimplemented(
        "GetDefaultDeviceAssignment not available with compile-only client.");
  }
  StatusOr<ifrt::Device*> LookupDevice(int device_id) const override {
    return Unimplemented(
        "LookupDevice not available with compile-only client.");
  }

  StatusOr<ifrt::ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return Unimplemented(
        "CreateDeviceToHostChannelHandle not available with compile-only "
        "client.");
  }
  StatusOr<ifrt::ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return Unimplemented(
        "CreateHostToDeviceChannelHandle not available with compile-only "
        "client.");
  }

  ifrt::Compiler* GetDefaultCompiler() override { return &default_compiler_; }

  static char ID;  // NOLINT

  const PjRtTopologyDescription& topology() const { return *topology_; }

 private:
  InvalidIfrtCompiler default_compiler_;
  std::shared_ptr<PjRtTopologyDescription> topology_;
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> descriptions_;
  std::vector<std::unique_ptr<PjRtCompileOnlyDevice>> owned_devices_;
  std::vector<PjRtDevice*> devices_;
};

char CompileOnlyIfRtClient::ID = 0;

class CompileOnlyPyClient : public PyClient {
 public:
  using PyClient::PyClient;

  StatusOr<std::shared_ptr<PjRtExecutable>> CompileUnloaded(
      std::string mlir_module, CompileOptions options,
      std::vector<pybind11::capsule> host_callbacks) {
    if (!host_callbacks.empty()) {
      return Unimplemented(
          "Compiling with host_callbacks not available with compile-only "
          "client.");
    }
    pybind11::gil_scoped_release gil_release;
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
};

}  // namespace

std::shared_ptr<PyClient> MakeCompileOnlyClient(
    std::shared_ptr<PjRtTopologyDescription> topology) {
  return std::make_shared<CompileOnlyPyClient>(
      std::make_unique<CompileOnlyIfRtClient>(std::move(topology)));
}

void RegisterCompileOnlyClient(pybind11::module& m) {
  pybind11::class_<CompileOnlyPyClient, PyClient,
                   std::shared_ptr<CompileOnlyPyClient>>(m,
                                                         "CompileOnlyPyClient")
      .def("compile",
           xla::ValueOrThrowWrapper(&CompileOnlyPyClient::CompileUnloaded),
           pybind11::arg("computation"),
           pybind11::arg("compile_options") = CompileOptions(),
           pybind11::arg("host_callbacks") = std::vector<pybind11::capsule>());
}

}  // namespace xla
