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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_FUSION_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_FUSION_THUNK_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/onednn_fusion.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// oneDNN fusion thunk encapsulates oneDNN graph constructed from an XLA fusion
// operation, where each HLO op has a corresponding oneDNN operation.
class OneDnnFusionThunk : public Thunk {
 public:
  ~OneDnnFusionThunk() override;

  struct Argument {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  struct Result {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  // Builder function constructs oneDNN fusion for the fusion operation.
  using Builder = absl::AnyInvocable<absl::StatusOr<OneDnnFusion>(
      absl::Span<const Argument> arguments, absl::Span<const Result> results)>;

  static absl::StatusOr<std::unique_ptr<OneDnnFusionThunk>> Create(
      Info info, std::vector<Argument> arguments, std::vector<Result> results,
      Builder builder);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 protected:
  OneDnnFusionThunk(Info info, std::vector<Argument> arguments,
                    std::vector<Result> results, Builder builder);

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
  // oneDNN runtime instantiated for the fusion operation.
  struct OneDnnRuntime;

  absl::StatusOr<OneDnnRuntime> CreateOneDnnRuntime(
      const Eigen::ThreadPoolDevice* device,
      absl::FunctionRef<absl::StatusOr<OneDnnFusion>()> builder);

  std::vector<Argument> arguments_;
  std::vector<Result> results_;

  Builder builder_;

  // XLA:CPU executable can be called concurrently from multiple threads,
  // and we need to keep a pool of oneDNN runtimes to avoid data races.
  ObjectPool<OneDnnRuntime, const Eigen::ThreadPoolDevice*>
      onednn_runtime_pool_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_ONEDNN_FUSION_THUNK_H_
