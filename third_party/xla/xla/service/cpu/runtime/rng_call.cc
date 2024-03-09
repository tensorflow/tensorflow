// Copyright 2023 The OpenXLA Authors.
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

#include "xla/service/cpu/runtime/rng_call.h"

#include <array>
#include <cstdint>

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/executable_run_options.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/cpu/runtime/rng.h"

namespace xla {
namespace cpu {

using ::xla::runtime::CustomCall;
using ::xla::runtime::Executable;
using ::xla::runtime::FlatMemrefView;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

static bool ThreeFry(xla::runtime::ExecutionContext* ctx, void** args,
                     void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla_cpu_rng_three_fry")
          .UserData<const ExecutableRunOptions*>()
          .Arg<FlatMemrefView>()
          .Arg<FlatMemrefView>()
          .Arg<FlatMemrefView>()
          .To<RuntimeChecks()>(xla::cpu::XlaThreeFry::Handler())
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

static bool Philox(xla::runtime::ExecutionContext* ctx, void** args,
                   void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla_cpu_rng_philox")
          .UserData<const ExecutableRunOptions*>()
          .Arg<FlatMemrefView>()
          .Arg<FlatMemrefView>()
          .Arg<FlatMemrefView>()
          .To<RuntimeChecks()>(xla::cpu::XlaPhilox::Handler())
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void PopulateXlaCpuRngCall(xla::runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla_cpu_rng_three_fry", &ThreeFry);
  registry.Register("xla_cpu_rng_philox", &Philox);
}

}  // namespace cpu
}  // namespace xla
