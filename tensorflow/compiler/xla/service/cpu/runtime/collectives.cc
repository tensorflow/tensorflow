// Copyright 2022 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/xla/service/cpu/runtime/collectives.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"

namespace xla {
namespace cpu {

using mlir::succeeded;

using ::xla::runtime::CustomCall;
using ::xla::runtime::Executable;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaPartitionId {
  absl::StatusOr<int32_t> operator()(
      const ExecutableRunOptions* run_options) const;
  static XlaPartitionId Handler() { return XlaPartitionId(); }
};
}  // namespace

absl::StatusOr<int32_t> XlaPartitionId::operator()(
    const ExecutableRunOptions* run_options) const {
  int32_t result;
  __xla_cpu_runtime_PartitionId(run_options, &result);
  return result;
}

static bool PartitionId(xla::runtime::ExecutionContext* ctx, void** args,
                        void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.partition_id")
                             .Ret<int32_t>()
                             .UserData<const ExecutableRunOptions*>()
                             .To<RuntimeChecks()>(XlaPartitionId::Handler())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaReplicaId {
  absl::StatusOr<int32_t> operator()(
      const ExecutableRunOptions* run_options) const;
  static XlaReplicaId Handler() { return XlaReplicaId(); }
};
}  // namespace

absl::StatusOr<int32_t> XlaReplicaId::operator()(
    const ExecutableRunOptions* run_options) const {
  int32_t result;
  __xla_cpu_runtime_ReplicaId(run_options, &result);
  return result;
}

static bool ReplicaId(xla::runtime::ExecutionContext* ctx, void** args,
                      void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.replica_id")
                             .Ret<int32_t>()
                             .UserData<const ExecutableRunOptions*>()
                             .To<RuntimeChecks()>(XlaReplicaId::Handler())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- /

void PopulateXlaCpuCollectivesCall(
    xla::runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.cpu.partition_id", &xla::cpu::PartitionId);
  registry.Register("xla.cpu.replica_id", &xla::cpu::ReplicaId);
}

}  // namespace cpu
}  // namespace xla
