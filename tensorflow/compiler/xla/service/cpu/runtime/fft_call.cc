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
#include "tensorflow/compiler/xla/service/cpu/runtime/fft_call.h"

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

#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fft.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla.pb.h"

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

namespace {
struct XlaFft {
  absl::Status operator()(const ExecutableRunOptions* run_options,
                          MemrefView input, MemrefView output, int32_t fft_type,
                          absl::Span<const int64_t> fft_length) const;
  static XlaFft Handler() { return XlaFft(); }
};
}  // namespace

absl::Status XlaFft::operator()(const ExecutableRunOptions* run_options,
                                MemrefView input, MemrefView output,
                                int32_t fft_type,
                                absl::Span<const int64_t> fft_length) const {
  if (fft_length.empty() || fft_length.size() > 3) {
    return absl::InvalidArgumentError(
        "fft_length must contain 1 to 3 elements");
  }
  bool double_precision = output.dtype == PrimitiveType::C128;
  int64_t input_batch = 1;
  auto fft_rank = static_cast<int32_t>(fft_length.size());
  for (int64_t dim = 0; dim < input.sizes.size() - fft_rank; ++dim) {
    input_batch *= input.sizes[dim];
  }
  __xla_cpu_runtime_EigenFft(run_options, output.data, input.data, fft_type,
                             static_cast<int32_t>(double_precision), fft_rank,
                             input_batch, fft_length[0],
                             fft_length.size() > 1 ? fft_length[1] : 0,
                             fft_length.size() > 2 ? fft_length[2] : 0);
  return absl::OkStatus();
}

static bool Fft(xla::runtime::ExecutionContext* ctx, void** args, void** attrs,
                void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.fft")
                             .UserData<const ExecutableRunOptions*>()
                             .Arg<MemrefView>()  // input
                             .Arg<MemrefView>()  // output
                             .Attr<int32_t>("fft_type")
                             .Attr<absl::Span<const int64_t>>("fft_length")
                             .To<RuntimeChecks()>(XlaFft::Handler())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void PopulateXlaCpuFftCall(xla::runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.cpu.fft", &Fft);
}

}  // namespace cpu
}  // namespace xla
