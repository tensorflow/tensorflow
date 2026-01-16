/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_FUSION_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_FUSION_THUNK_H_

#include <stdbool.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// YNN fusion thunk encapsulates YNNPACK subgraph constructed from an XLA fusion
// operation.
class YnnFusionThunk : public Thunk {
 public:
  ~YnnFusionThunk() override;

  struct Options {
    // Pass YnnThreadpool constructed from the intra-op threadpool to the
    // YNNPACK runtime to allow YNNPACK to parallelize the execution.
    bool use_threadpool = true;
  };

  struct Argument {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  struct Result {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  // Builder function constructs YNNPACK subgraph for the fusion operation.
  using Builder = absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
      absl::Span<const Argument> arguments, absl::Span<const Result> results)>;

  // Builder function that constructs YNNPACK subgraph for the fusion operation
  // and captures some of the arguments buffers by value. Such YNNPACK subgraphs
  // can't be reused if captured arguments are not the same, and can lead to
  // crashes and undefined behavior if captured arguments are destroyed.
  // Capturing arguments by value allows YNNPACK to do packing at graph compile
  // time, and avoid re-packing costs at run time (at inference weights stay
  // constant, i.e. convolution filters and one of the dot arguments).
  using CapturingBuilder = absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
      absl::Span<const Argument> arguments, absl::Span<const Result> results,
      absl::Span<const se::DeviceAddressBase> arguments_buffers)>;

  static absl::StatusOr<std::unique_ptr<YnnFusionThunk>> Create(
      Options options, Info info, const HloInstruction* hlo,
      std::vector<Argument> arguments, std::vector<Result> results,
      Builder builder);

  static absl::StatusOr<std::unique_ptr<YnnFusionThunk>> Create(
      Options options, Info info, const HloInstruction* hlo,
      std::vector<Argument> arguments, std::vector<Result> results,
      CapturingBuilder capturing_builder,
      absl::Span<const int64_t> captured_arguments_ids);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  bool ExecuteMayBlock() const final { return true; }

  BufferUses buffer_uses() const final;

  Options options() const { return options_; }

  const HloInstruction* hlo() const { return hlo_; }

  absl::Span<const Argument> arguments() const { return arguments_; }
  absl::Span<const Result> results() const { return results_; }

 protected:
  YnnFusionThunk(Options options, Info info, const HloInstruction* hlo,
                 std::vector<Argument> arguments, std::vector<Result> results,
                 Builder builder);

  YnnFusionThunk(Options options, Info info, const HloInstruction* hlo,
                 std::vector<Argument> arguments, std::vector<Result> results,
                 CapturingBuilder capturing_builder,
                 absl::Span<const int64_t> captured_arguments_ids);

  // Extension points for subclasses to customize the logging behavior.
  virtual std::string fusion_kind() const { return "fusion"; }
  virtual std::string fusion_description() const { return ""; }

  virtual bool has_fusion_details() const { return false; }
  virtual std::vector<std::string> fusion_details() const { return {}; }

  virtual std::string argument_name(size_t index) const {
    return absl::StrCat("arg #", index);
  }

  virtual std::string result_name(size_t index) const {
    return absl::StrCat("res #", index);
  }

 private:
  // YNNPACK subgraph + runtime instantiated and ready for execution.
  struct YnnExecutable;

  // Creates YnnExecutable for the fusion operation using one of the builders.
  absl::StatusOr<YnnExecutable> CreateYnnExecutable(
      const YnnThreadpool& threadpool,
      absl::Span<const se::DeviceAddressBase> arguments_buffers);

  // Updates YnnExecutable to the YNN subgraph constructed with the given
  // arguments buffers.
  absl::Status UpdateYnnExecutable(
      const YnnThreadpool& threadpool, YnnExecutable& executable,
      absl::Span<const se::DeviceAddressBase> arguments_buffers);

  // Returns the list of captured arguments buffers.
  std::vector<se::DeviceAddressBase> CaptureArguments(
      absl::Span<const se::DeviceAddressBase> arguments_buffers);

  Options options_;

  // A pointer to the HLO instruction that this thunk is associated with. Owned
  // by the `HloModule` associated with the XLA executable.
  const HloInstruction* hlo_;  // not owned

  std::vector<Argument> arguments_;
  std::vector<Result> results_;

  // Builder that constructs YNNPACK subgraph for the fusion operation.
  Builder builder_;

  // Builder that constructs YNNPACK subgraph for the fusion operation and
  // captures some of the arguments buffers by value. Such subgraphs can't be
  // reused if captured arguments changed since the last execution.
  CapturingBuilder capturing_builder_;

  // Indices of arguments that are captured by YNNPACK subgraph by value.
  std::vector<int64_t> captured_arguments_ids_;

  // XLA:CPU executable can be called concurrently from multiple threads,
  // and we need to keep a pool of YNNPACK executables to avoid data races.
  using YnnExecutablePool = ObjectPool<YnnExecutable, const YnnThreadpool&,
                                       absl::Span<const se::DeviceAddressBase>>;
  YnnExecutablePool ynn_executable_pool_;

  // The number of YNNPACK executables created for capturing graphs.
  std::atomic<int64_t> num_capturing_created_{0};
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_YNNPACK_YNN_FUSION_THUNK_H_
