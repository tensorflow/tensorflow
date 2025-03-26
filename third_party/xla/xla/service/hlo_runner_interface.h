/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_RUNNER_INTERFACE_H_
#define XLA_SERVICE_HLO_RUNNER_INTERFACE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_module_util.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

class HloRunnerInterface;
class BufferAssignmentProto;

// Tags to identify particular properties of a HloRunnerInterface
// implementation.
//
// Tags are an opaque way to expose arbitrary details about the runner backend.
// Tags should be added whenever a decision must be made based on the property
// of a particular backend or runner implementation.
//
// For example, if a specific feature is only supported under certain conditions
// and only known to the backend, a tag can be added here and exposed via all
// applicable backends. A test or other functionality can then be gated on the
// presence of that particular tag.
//
// Custom tags that cannot be added in this file can be defined elsewhere but
// should use negative values to avoid conflicts.
class HloRunnerPropertyTag final {
 public:
  // Underlying type for HloRunnerPropertyTag properties as well as other fields
  // that represent property tags.
  //
  // e.g. a custom grouping class for a proprietary codebase could be defined
  // as:
  //
  // class MyCorpPropertyTag final {
  //  public:
  //   static constexpr HloRunnerPropertyTag::Type kInternalFeature1 = -1;
  //   static constexpr HloRunnerPropertyTag::Type kInternalFeature2 = -2;
  // };
  using Type = int;

  // Default, reserved value for HloRunnerPropertyTag. Perhaps this could be
  // used as a sentinel value for a tag that is not present. Do not use.
  static constexpr Type kDefault = 0;
  // Indicates that the runner is using ROCm.
  static constexpr Type kUsingGpuRocm = 1;
  // Indicates that this runner is a CPU runner.
  static constexpr Type kCpu = 2;

 private:
  HloRunnerPropertyTag() = default;
};

// Runner implementations only support the execution of executables that were
// created by the same runner. We use the this class to represent these
// executables when they leave the runner without exposing any details of the
// underlying implementation. See go/xla-opaque-executable for more details.
class OpaqueExecutable {
 public:
  virtual ~OpaqueExecutable() = default;

  // !!! STOP !!!
  // Before adding any methods to this class, please consider if they could be
  // added to the HloRunnerInterface instead.
  //
  // Adding methods to this class imposes a burden on runners as they must
  // implement and support any/all types used in the signature. The runner
  // itself should serve as the only means of accessing information about the
  // executable, since only the runner is capable of unwrapping the executable.
  //
  // E.g. you might be inclined to add a method to this class that returns a
  // HloModule. DON'T. Not all executables may have a HloModule, while some may
  // even have multiple. The runner interface has a HloModuleFromWrapped method
  // that has the semantics of returning the first HloModule in the executable
  // if there are multiple, or the sole HloModule if there is only one.
  // !!! STOP !!!

 protected:
  explicit OpaqueExecutable(absl::Nonnull<const HloRunnerInterface*> creator)
      : creator_(ABSL_DIE_IF_NULL(creator)) {}
  // Cannot be moved or copied.
  OpaqueExecutable(const OpaqueExecutable&) = default;
  OpaqueExecutable& operator=(const OpaqueExecutable&) = default;

  template <typename T>
  static absl::StatusOr<absl::Nonnull<T*>> TryUnwrap(
      const HloRunnerInterface& runner,
      absl::Nonnull<OpaqueExecutable*> const wrapped) {
    static_assert(
        std::is_base_of_v<OpaqueExecutable, T>,
        "TryUnwrap must be used with a subclass of OpaqueExecutable.");
    if (wrapped->creator_ != &runner) {
      return absl::InvalidArgumentError(
          "Executable was not created by this runner.");
    }

    if (T* const executable = tensorflow::down_cast<T*>(wrapped);
        executable != nullptr) {
      return executable;
    }
    return absl::InvalidArgumentError("Invalid opaque executable.");
  }

  template <typename T>
  static absl::StatusOr<absl::Nonnull<const T*>> TryUnwrap(
      const HloRunnerInterface& runner,
      absl::Nonnull<const OpaqueExecutable*> const wrapped) {
    static_assert(
        std::is_base_of_v<OpaqueExecutable, T>,
        "TryUnwrap must be used with a subclass of OpaqueExecutable.");
    if (wrapped->creator_ != &runner) {
      return absl::InvalidArgumentError(
          "Executable was not created by this runner.");
    }

    if (const T* const executable = tensorflow::down_cast<const T*>(wrapped);
        executable != nullptr) {
      return executable;
    }
    return absl::InvalidArgumentError("Invalid opaque executable.");
  }

  const HloRunnerInterface* const creator_;
};

// A base class for running an HloModule. This executes the given HloModule on a
// certain backend directly without using the client interface. HloModule can be
// explicitly built, or loaded from a serialization file (e.g., hlo proto
// file), or parsed from a hlo textual IR string.
class HloRunnerInterface {
 public:
  // The options used to configure an ExecuteReplicated() call.
  struct ReplicatedExecuteOptions {
    // The number of devices the HLO module should be replicated onto.
    int64_t num_replicas = 1;

    // The arguments to be fed to each replica. Since this is used for a
    // replicated execution, all the arguments are the same for all replicas.
    std::vector<const Literal*> arguments;

    // If the HLO module being run has an infeed instruction, this will be the
    // data which will be fed to it, for as many as infeed_steps steps.
    std::vector<const Literal*> infeed_values;

    // The number of times the infeed literal should be fed to the HLO module.
    // For a clean exit, this should match the iterations-per-loop parameter
    // used when generating the HLO module proto (that is usually the main
    // while boundary counter). A value higher then iterations-per-loop would
    // lead to infeed threads feeding to a gone computation, while a lower
    // value would trigger a stuck ExecuteReplicated() call (the computation
    // will be trying to infeed data which will never come).
    int64_t infeed_steps = -1;

    // The shape of the outfeed operation. If empty, the HLO module does not
    // generate any outfeed.
    Shape outfeed_shape;

    // A pointer to a vector where the outfeed values will be stored. If
    // nullptr, the values will be read and discarded.
    std::vector<Literal>* outfeed_values = nullptr;

    // Whether the HLO passes should be run on the input module. Usually
    // saved modules are coming from after the HLO pass pipeline, so triggering
    // another run will likely cause errors.
    bool run_hlo_passes = false;

    // If true, executes on multiple threads using se::Stream::ExecuteOnStream.
    // Otherwise, executes using xla::Executable::ExecuteOnStreams.
    bool use_threads = false;
  };

  using DeviceShapeRepresentationFn = std::function<Shape(const Shape&)>;
  using DeviceShapeSizeFn = std::function<int64_t(const Shape&)>;

  HloRunnerInterface() = default;
  virtual ~HloRunnerInterface() = default;

  // Creates a runner-internal executable object given an HLO module and returns
  // a OpaqueExecutable. If run_hlo_passes is true, the HLO passes will be run
  // as part of compilation.
  virtual absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) = 0;

  // Creates a runner-internal executable object given a runner and
  // platform-specific serialized executable representation. The serialized
  // representation must have been produced by a compiler of the same platform
  // and version as this one.
  virtual absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
  DeserializeExecutable(
      absl::Nonnull<const tsl::protobuf::Message*> serialized) const = 0;

  // Same as above, except it takes buffer assignment as input.
  // Note: The default implementation of the API here does not utilize the given
  // buffer assignment. A derived runner interface is expected to override the
  // following method to achieve this functionality.
  virtual absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
  CreateExecutableWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* /*buffer_assignment_proto*/,
      bool run_hlo_passes) {
    LOG(WARNING) << "Ignoring the buffer assignment proto provided.";
    return CreateExecutable(std::move(module), run_hlo_passes);
  }

  // Executes the given module with given literals as input and returns the
  // result as a Literal.
  //
  // If run_hlo_passes is false, the module will be executed without Hlo
  // optimization
  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal* const> arguments,
                                  bool run_hlo_passes = true) {
    return Execute(std::move(module), arguments, run_hlo_passes, nullptr);
  }

  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal> arguments,
                                  bool run_hlo_passes = true,
                                  ExecutionProfile* profile = nullptr);

  virtual absl::StatusOr<Literal> Execute(
      std::unique_ptr<HloModule> module,
      absl::Span<const Literal* const> arguments, bool run_hlo_passes,
      ExecutionProfile* profile) = 0;

  // Same as above 3 methods, but with buffer assignment specified.
  absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      absl::Span<const Literal* const> arguments, bool run_hlo_passes = true) {
    return ExecuteWithBufferAssignment(std::move(module),
                                       buffer_assignment_proto, arguments,
                                       run_hlo_passes, nullptr);
  }

  absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      absl::Span<const Literal> arguments, bool run_hlo_passes = true,
      ExecutionProfile* profile = nullptr);

  // Note: The default implementation of the API here does not utilize the given
  // buffer assignment. A derived runner interface is expected to override the
  // following method to achieve this functionality.
  virtual absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* /*buffer_assignment_proto*/,
      absl::Span<const Literal* const> arguments, bool run_hlo_passes,
      ExecutionProfile* profile) {
    LOG(WARNING) << "Ignoring the buffer assignment proto provided.";
    return Execute(std::move(module), arguments, run_hlo_passes, profile);
  }

  // Same as 3 Execute methods above, but with Executable as input.
  absl::StatusOr<Literal> ExecuteWithExecutable(
      OpaqueExecutable* executable, absl::Span<const Literal> arguments,
      ExecutionProfile* profile = nullptr);

  absl::StatusOr<Literal> ExecuteWithExecutable(
      OpaqueExecutable* executable,
      absl::Span<const Literal* const> arguments) {
    return ExecuteWithExecutable(executable, arguments, nullptr);
  }

  virtual absl::StatusOr<Literal> ExecuteWithExecutable(
      OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
      ExecutionProfile* profile) = 0;

  // Executes a given HLO module into a set of replicas, and returns a map
  // with the replica number as key, and the corresponding returned literal as
  // value.
  // TODO(b/172931928): change to non-virtual function.
  virtual absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) = 0;

  // Same as above, but with specified device assignment.
  virtual absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) = 0;

  virtual absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<OpaqueExecutable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) = 0;

  // Returns the name of this runner.
  virtual absl::string_view Name() const = 0;

  // Return the device shape representation of 'host_shape'.
  virtual DeviceShapeRepresentationFn device_shape_representation_fn()
      const = 0;
  // Return the device shape size of 'host_shape'.
  // This function is used e.g. to create a VerifiedHloModule. It returns an
  // integer representing the size of the shape in bytes as opposed to a Shape.
  virtual DeviceShapeSizeFn device_shape_size_fn() const = 0;

  // Returns the number of devices which are known. Not all of these devices may
  // be usable by XLA.
  virtual int device_count() const = 0;

  // Returns true if the condition corresponding to the given tag is true for
  // this runner.
  virtual bool HasProperty(HloRunnerPropertyTag::Type tag) const = 0;

  // Returns the first (or only) HloModule associated with the given
  // OpaqueExecutable. Returns an error if the OpaqueExecutable cannot be
  // unwrapped, or if the OpaqueExecutable does not contain at least one
  // HloModule.
  virtual absl::StatusOr<absl::Nonnull<const HloModule*>> HloModuleFromWrapped(
      const OpaqueExecutable* wrapped) const = 0;

  // Returns true if the two given OpaqueExecutables originate from the same
  // runner and are equivalent according to some notion specific to that runner.
  // Executables that were created by different runners can never be equivalent.
  virtual bool ExecutablesAreEquivalent(
      absl::Nonnull<const OpaqueExecutable*> lhs,
      absl::Nonnull<const OpaqueExecutable*> rhs) const = 0;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_RUNNER_INTERFACE_H_
