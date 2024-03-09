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
#include "xla/service/cpu/runtime/convolution_call.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/executable_run_options.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/cpu/runtime/convolution.h"

namespace xla {
namespace cpu {

using ::xla::runtime::CustomCall;
using ::xla::runtime::Executable;
using ::xla::runtime::MemrefView;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

static bool Convolution(xla::runtime::ExecutionContext* ctx, void** args,
                        void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("xla_cpu_convolution")
          .UserData<const ExecutableRunOptions*>()
          .Arg<MemrefView>()  // input
          .Arg<MemrefView>()  // kernel
          .Arg<MemrefView>()  // output
          .Attr<int64_t>("inputBatchDimension")
          .Attr<absl::Span<const int64_t>>("inputSpatialDimensions")
          .Attr<int64_t>("inputFeatureDimension")
          .Attr<absl::Span<const int64_t>>("kernelSpatialDimensions")
          .Attr<int64_t>("kernelInputFeatureDimension")
          .Attr<int64_t>("kernelOutputFeatureDimension")
          .Attr<absl::Span<const int64_t>>("outputSpatialDimensions")
          .Attr<absl::Span<const int64_t>>("window_strides")
          .Attr<absl::Span<const int64_t>>("padding")
          .Attr<absl::Span<const int64_t>>("lhs_dilation")
          .Attr<absl::Span<const int64_t>>("rhs_dilation")
          .Attr<int64_t>("feature_group_count")
          .To<RuntimeChecks()>(xla::cpu::XlaConvolution::Handler())
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void PopulateXlaCpuConvolutionCall(
    xla::runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla_cpu_convolution", &Convolution);
}

}  // namespace cpu
}  // namespace xla
