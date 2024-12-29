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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/layout.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/py_client.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {

namespace {

class CompileOnlyMemory
    : public llvm::RTTIExtends<CompileOnlyMemory, ifrt::Memory> {
 public:
  explicit CompileOnlyMemory(
      int id, const PjRtMemorySpaceDescription* memory_description,
      ifrt::Device* device)
      : id_(id),
        kind_(memory_description->kind()),
        debug_string_(absl::StrFormat("CompileOnlyMemory(id=%d, kind=%s)", id,
                                      memory_description->kind())),
        device_(device) {}

  ifrt::MemoryId Id() const override { return ifrt::MemoryId(id_); }

  const ifrt::MemoryKind& Kind() const override { return kind_; }

  absl::string_view ToString() const override { return debug_string_; }
  absl::string_view DebugString() const override { return debug_string_; }

  absl::Span<ifrt::Device* const> Devices() const override {
    return absl::Span<ifrt::Device* const>{&device_, 1};
  }

  static char ID;  // NOLINT

 private:
  int id_;
  ifrt::MemoryKind kind_;
  std::string debug_string_;
  ifrt::Device* device_;
};

[[maybe_unused]] char CompileOnlyMemory::ID = 0;

class CompileOnlyDevice
    : public llvm::RTTIExtends<CompileOnlyDevice, ifrt::Device> {
 public:
  explicit CompileOnlyDevice(const PjRtDeviceDescription* description)
      : description_(std::move(description)),
        attributes_(ifrt::FromPjRtAttributeMap(description_->Attributes())) {}

  const PjRtDeviceDescription& description() const { return *description_; }

  ifrt::Client* client() const override { return nullptr; }
  bool IsAddressable() const override { return false; }
  ifrt::DeviceId Id() const override {
    return ifrt::DeviceId(description_->id());
  }

  int ProcessIndex() const override { return description_->process_index(); }

  absl::string_view Kind() const override {
    return description_->device_kind();
  }

  absl::string_view ToString() const override {
    return description_->ToString();
  }

  absl::string_view DebugString() const override {
    return description_->DebugString();
  }

  absl::Span<ifrt::Memory* const> Memories() const override {
    return unowned_memories_;
  }
  absl::StatusOr<ifrt::Memory*> DefaultMemory() const override {
    if (default_memory_) {
      return default_memory_;
    }
    return Unimplemented("DefaultMemory is not supported");
  }

  const ifrt::AttributeMap& Attributes() const override { return attributes_; }

  void AttachMemory(std::unique_ptr<ifrt::Memory> memory) {
    unowned_memories_.push_back(memory.get());
    owned_memories_.push_back(std::move(memory));
  }

  void SetDefaultMemory(ifrt::Memory* memory) { default_memory_ = memory; }

 private:
  const PjRtDeviceDescription* description_;
  ifrt::AttributeMap attributes_;
  ifrt::Memory* default_memory_ = nullptr;
  std::vector<ifrt::Memory*> unowned_memories_;
  std::vector<std::unique_ptr<ifrt::Memory>> owned_memories_;
};

class InvalidIfrtCompiler final
    : public llvm::RTTIExtends<InvalidIfrtCompiler, ifrt::Compiler> {
 public:
  absl::StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> Compile(
      std::unique_ptr<ifrt::Program> program,
      std::unique_ptr<ifrt::CompileOptions> options) override {
    return Unimplemented("Compile not implemented.");
  }

  absl::StatusOr<std::unique_ptr<ifrt::Executable>> Compile(
      std::unique_ptr<ifrt::Program> program, const ifrt::Topology& topology,
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
[[maybe_unused]] char InvalidIfrtCompiler::ID = 0;

class CompileOnlyIfRtClient final
    : public llvm::RTTIExtends<CompileOnlyIfRtClient, ifrt::Client> {
 public:
  explicit CompileOnlyIfRtClient(std::shared_ptr<ifrt::PjRtTopology> topology)
      : topology_(std::move(topology)),
        descriptions_(topology_->DeviceDescriptions()),
        attributes_(ifrt::AttributeMap::Map()) {
    int offset = 0;
    for (auto& description : descriptions_) {
      owned_devices_.push_back(
          std::make_unique<CompileOnlyDevice>(description.get()));
      auto* device = owned_devices_.back().get();
      devices_.push_back(device);
      if (description->process_index() == process_index()) {
        auto default_memory = description->default_memory_space();
        for (auto* memory_description : description->memory_spaces()) {
          auto memory = std::make_unique<CompileOnlyMemory>(
              offset, memory_description, device);
          if (default_memory.ok() && memory_description == *default_memory) {
            device->SetDefaultMemory(memory.get());
          }
          device->AttachMemory(std::move(memory));
          ++offset;
        }
      }
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
  absl::StatusOr<tsl::RCReference<ifrt::Array>>
  AssembleArrayFromSingleDeviceArrays(
      ifrt::Shape shape, std::shared_ptr<const ifrt::Sharding> sharding,
      absl::Span<tsl::RCReference<ifrt::Array>> arrays,
      ifrt::ArrayCopySemantics array_copy_semantics,
      ifrt::SingleDeviceShardSemantics single_device_shard_semantics) override {
    return Unimplemented(
        "AssembleArrayFromSingleDeviceArrays not available with compile-only "
        "client.");
  }

  absl::StatusOr<std::vector<tsl::RCReference<ifrt::Array>>> CopyArrays(
      absl::Span<tsl::RCReference<ifrt::Array>> arrays,
      std::optional<tsl::RCReference<ifrt::DeviceList>> devices,
      std::optional<ifrt::MemoryKind> memory_kind,
      ifrt::ArrayCopySemantics semantics) override {
    return Unimplemented("CopyArrays not available with compile-only client.");
  }

  absl::StatusOr<std::vector<tsl::RCReference<ifrt::Array>>> RemapArrays(
      const ifrt::RemapPlan& plan,
      absl::Span<tsl::RCReference<ifrt::Array>> arrays,
      ifrt::ArrayCopySemantics semantics) override {
    return Unimplemented("RemapArrays not available with compile-only client.");
  }

  ifrt::Future<> GetReadyFuture(
      absl::Span<const tsl::RCReference<ifrt::Value>> values) override {
    return ifrt::Future<>(Unimplemented(
        "GetReadyFuture not available with compile-only client."));
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
  const ifrt::AttributeMap& Attributes() const override { return attributes_; }

  int device_count() const override { return devices().size(); }
  int addressable_device_count() const override { return 0; }
  absl::Span<ifrt::Device* const> devices() const override { return devices_; }
  absl::Span<ifrt::Device* const> addressable_devices() const override {
    return {};
  }
  int process_index() const override { return 0; }
  absl::Span<xla::ifrt::Device* const> GetAllDevices() const override {
    return devices_;
  }
  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return Unimplemented(
        "GetDefaultDeviceAssignment not available with compile-only client.");
  }
  absl::StatusOr<ifrt::Device*> LookupDevice(
      ifrt::DeviceId device_id) const override {
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

  const ifrt::PjRtTopology& topology() const { return *topology_; }

  absl::StatusOr<std::shared_ptr<ifrt::Topology>> GetTopologyForDevices(
      const tsl::RCReference<xla::ifrt::DeviceList>& devices) const override {
    return topology_;
  }

  absl::StatusOr<std::unique_ptr<PjRtLayout>> GetDefaultLayoutForDevice(
      ifrt::DType dtype, absl::Span<const int64_t> dims,
      ifrt::Device* device) const override {
    TF_ASSIGN_OR_RETURN(PrimitiveType element_type, ToPrimitiveType(dtype));
    TF_ASSIGN_OR_RETURN(xla::Layout layout,
                        topology_->GetDefaultLayout(element_type, dims));
    return std::make_unique<PjRtXlaLayout>(std::move(layout));
  }

 private:
  InvalidIfrtCompiler default_compiler_;
  std::shared_ptr<ifrt::PjRtTopology> topology_;
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> descriptions_;
  ifrt::AttributeMap attributes_;
  std::vector<std::unique_ptr<CompileOnlyDevice>> owned_devices_;
  std::vector<ifrt::Device*> devices_;
};

[[maybe_unused]] char CompileOnlyIfRtClient::ID = 0;

class CompileOnlyPyClient : public PyClient {
 public:
  using PyClient::PyClient;

  static nb_class_ptr<PyClient> Make(
      std::shared_ptr<ifrt::PjRtTopology> topology) {
    auto client =
        nb::borrow<nb_class_ptr<PyClient>>(make_nb_class<CompileOnlyPyClient>(
            std::make_unique<CompileOnlyIfRtClient>(std::move(topology))));
    CompileOnlyPyClient::Initialize(client);
    return client;
  }

  absl::StatusOr<std::shared_ptr<ifrt::Executable>> CompileUnloaded(
      absl::string_view mlir_module, CompileOptions options,
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
    if (options.executable_build_options.use_shardy_partitioner()) {
      // Since Shardy is located in the middle of the XLA pipeline, we need to
      // export it before going to HLO while preserving Shardy ops and attrs.
      TF_RETURN_IF_ERROR(ExportShardyForHloRoundTrip(*module));
    }
    auto* ifrt_client =
        llvm::dyn_cast_or_null<CompileOnlyIfRtClient>(this->ifrt_client());
    CHECK(ifrt_client) << "CompileOnlyPyClient requires ifrt_client be a "
                          "CompileOnlyIfRtClient";
    auto xla_options = std::make_unique<ifrt::XlaCompileOptions>(options);
    TF_ASSIGN_OR_RETURN(auto executable,
                        PjRtCompile(std::move(options), module.get(),
                                    *ifrt_client->topology().description()));
    TF_ASSIGN_OR_RETURN(auto ifrt_executable,
                        ifrt::PjRtExecutable::Create(std::move(executable),
                                                     std::move(xla_options)));
    return std::shared_ptr<ifrt::Executable>(std::move(ifrt_executable));
  }

 private:
  static void Initialize(nb_class_ptr<PyClient> client) {
    PyClient::Initialize(client);
  }
};

}  // namespace

nb_class_ptr<PyClient> MakeCompileOnlyClient(
    std::shared_ptr<ifrt::PjRtTopology> topology) {
  return CompileOnlyPyClient::Make(std::move(topology));
}

void RegisterCompileOnlyClient(nb::module_& m) {
  nb::class_<CompileOnlyPyClient, PyClient>(m, "CompileOnlyPyClient")
      .def(
          "compile",
          [](CompileOnlyPyClient& self, nb::bytes mlir_module,
             CompileOptions options, std::vector<nb::capsule> host_callbacks) {
            return ValueOrThrow(self.CompileUnloaded(
                absl::string_view(mlir_module.c_str(), mlir_module.size()),
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
