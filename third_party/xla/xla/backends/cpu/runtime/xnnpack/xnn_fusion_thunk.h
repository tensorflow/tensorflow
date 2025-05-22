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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_FUSION_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_FUSION_THUNK_H_

#include <stdbool.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"

// Forward declare XNNPACK types.
typedef struct xnn_subgraph* xnn_subgraph_t;  // NOLINT

namespace xla::cpu {

// XNN fusion thunk encapsulates XNNPACK subgraph contructed from an XLA fusion
// operation, where each HLO op has a corresponding XNNPACK operator.
class XnnFusionThunk : public Thunk {
 public:
  enum class XnnFusionKind {
    kFusion,
    kDot,
    kConvolution,
  };

  static absl::string_view XnnFusionKindToString(XnnFusionKind kind);

  ~XnnFusionThunk() override;

  struct Options {
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

  // Builder function constructs XNNPACK subgraph for the fusion operation.
  using Builder = absl::AnyInvocable<absl::StatusOr<xnn_subgraph_t>(
      absl::Span<const Argument> arguments, absl::Span<const Result> results)>;

  // Builder function that constructs XNNPACK subgraph for the fusion operation
  // and captures some of the arguments buffers by value. Such XNNPACK subgraphs
  // can't be reused if captured arguments are not the same, and can lead to
  // crashes and undefined behavior if captured arguments are destroyed.
  // Capturing arguments by value allows XNNPACK to do packing at graph compile
  // time, and avoid re-packing costs at run time (at inference weights stay
  // constant, i.e. convolution filters and one of the dot arguments).
  using CapturingBuilder = absl::AnyInvocable<absl::StatusOr<xnn_subgraph_t>(
      absl::Span<const Argument> arguments, absl::Span<const Result> results,
      absl::Span<const se::DeviceMemoryBase> arguments_buffers)>;

  static absl::StatusOr<std::unique_ptr<XnnFusionThunk>> Create(
      Options options, Info info, std::vector<Argument> arguments,
      std::vector<Result> results, Builder builder);

  static absl::StatusOr<std::unique_ptr<XnnFusionThunk>> Create(
      Options options, Info info, std::vector<Argument> arguments,
      std::vector<Result> results, CapturingBuilder capturing_builder,
      absl::Span<const int64_t> captured_arguments);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

  Options options() const { return options_; }

  XnnFusionKind xnn_fusion_kind() const { return xnn_fusion_kind_; }

 protected:
  XnnFusionThunk(XnnFusionKind kind, Options options, Info info,
                 std::vector<Argument> arguments, std::vector<Result> results,
                 Builder builder);

  XnnFusionThunk(XnnFusionKind kind, Options options, Info info,
                 std::vector<Argument> arguments, std::vector<Result> results,
                 CapturingBuilder capturing_builder,
                 absl::Span<const int64_t> captured_arguments);

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
  // XNNPACK runtime instantiated for the fusion operation.
  struct XnnRuntime;

  absl::StatusOr<XnnRuntime> CreateXnnRuntime(
      const Eigen::ThreadPoolDevice* device, bool capturing,
      absl::FunctionRef<absl::StatusOr<xnn_subgraph_t>()> builder);

  Options options_;

  std::vector<Argument> arguments_;
  std::vector<Result> results_;

  // Only one kind of the builder should be set, depending on whether the
  // subgraph captures arguments by value.
  Builder builder_;
  CapturingBuilder capturing_builder_;

  XnnFusionKind xnn_fusion_kind_;

  // Indices of arguments that are captured by XNNPACK subgraph by value.
  absl::flat_hash_set<int64_t> captured_arguments_;

  // XLA:CPU executable can be called concurrently from multiple threads,
  // and we need to keep a pool of XNNPACK runtimes to avoid data races.
  ObjectPool<XnnRuntime, const Eigen::ThreadPoolDevice*> xnn_runtime_pool_;

  // The number of XNNPACK runtimes created for capturing graphs.
  std::atomic<int64_t> num_capturing_created_{0};
};

std::ostream& operator<<(std::ostream& os, XnnFusionThunk::XnnFusionKind kind);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_FUSION_THUNK_H_
