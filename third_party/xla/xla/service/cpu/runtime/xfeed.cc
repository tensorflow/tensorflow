// Copyright 2022 The OpenXLA Authors.
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

#include "xla/service/cpu/runtime/xfeed.h"

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

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/executable_run_options.h"
#include "xla/primitive_util.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

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

static xla::Shape ToShape(const xla::runtime::StridedMemrefView& memref) {
  // Recover `minor_to_major` dimensions permutation from strides.
  auto indexed_strides_range =
      llvm::map_range(llvm::enumerate(memref.strides), [](auto pair) {
        return std::pair<int64_t, int64_t>{pair.value(), pair.index()};
      });

  auto indexed_strides = llvm::to_vector(indexed_strides_range);
  llvm::stable_sort(indexed_strides);

  auto minor_to_major =
      llvm::to_vector(llvm::make_second_range(indexed_strides));
  return xla::ShapeUtil::MakeShapeWithDenseLayout(memref.dtype, memref.sizes,
                                                  minor_to_major);
}

static int64_t MemrefSize(const xla::runtime::StridedMemrefView& memref) {
  int64_t size_in_bytes = primitive_util::ByteWidth(memref.dtype);
  for (int64_t size : memref.sizes) {
    size_in_bytes *= size;
  }
  return size_in_bytes;
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaInfeed {
  absl::Status operator()(const ExecutableRunOptions* run_options,
                          CustomCall::RemainingArgs args) const;
  static XlaInfeed Handler() { return XlaInfeed(); }
};
}  // namespace

absl::Status XlaInfeed::operator()(const ExecutableRunOptions* run_options,
                                   CustomCall::RemainingArgs args) const {
  for (unsigned i = 0; i < args.size(); ++i) {
    auto memref = args.get<xla::runtime::StridedMemrefView>(i);
    if (!succeeded(memref)) {
      return absl::InvalidArgumentError(
          "Failed to get arguments as (strided) memref view");
    }

    auto size_in_bytes = static_cast<int32_t>(MemrefSize(*memref));
    std::string shape_string = ToShape(*memref).SerializeAsString();

    void* infeed_buffer = __xla_cpu_runtime_AcquireInfeedBufferForDequeue(
        run_options, size_in_bytes, shape_string.data(),
        static_cast<int32_t>(shape_string.size()));
    // Copy from the infeed buffer.
    std::memcpy(memref->data, infeed_buffer, size_in_bytes);
    __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(
        run_options, size_in_bytes, infeed_buffer, shape_string.data(),
        static_cast<int32_t>(shape_string.size()));
  }
  return absl::OkStatus();
}

static bool Infeed(xla::runtime::ExecutionContext* ctx, void** args,
                   void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.infeed")
                             .UserData<const ExecutableRunOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .To<RuntimeChecks()>(XlaInfeed::Handler())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

// -------------------------------------------------------------------------- //

namespace {
struct XlaOutfeed {
  absl::Status operator()(const ExecutableRunOptions* run_options,
                          CustomCall::RemainingArgs args,
                          absl::Span<const int32_t> result_type) const;
  static XlaOutfeed Handler() { return XlaOutfeed(); }
};
}  // namespace

absl::Status XlaOutfeed::operator()(
    const ExecutableRunOptions* run_options, CustomCall::RemainingArgs args,
    absl::Span<const int32_t> result_type) const {
  assert(result_type.size() == args.size() &&
         "Result types and input args should be of the same size.");
  for (unsigned i = 0; i < args.size(); ++i) {
    auto memref = args.get<xla::runtime::StridedMemrefView>(i);
    if (!succeeded(memref)) {
      return absl::InvalidArgumentError(
          "Failed to get arguments as (strided) memref view");
    }

    // Restoring the sign information that was lost during convert-to-signless
    // pass. This information was stashed in an attribute inside
    // xla_cpu::outfeed.
    memref->dtype = PrimitiveType(result_type[i]);

    auto size_in_bytes = static_cast<int32_t>(MemrefSize(*memref));
    std::string shape_string = ToShape(*memref).SerializeAsString();

    void* outfeed_buffer = __xla_cpu_runtime_AcquireOutfeedBufferForPopulation(
        run_options, size_in_bytes, shape_string.data(),
        static_cast<int32_t>(shape_string.size()));
    // Copy to the outfeed buffer.
    std::memcpy(outfeed_buffer, memref->data, size_in_bytes);
    __xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation(
        run_options, size_in_bytes, outfeed_buffer, shape_string.data(),
        static_cast<int32_t>(shape_string.size()));
  }
  return absl::OkStatus();
}

static bool Outfeed(xla::runtime::ExecutionContext* ctx, void** args,
                    void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.cpu.outfeed")
                             .UserData<const ExecutableRunOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .Attr<absl::Span<const int32_t>>("result_type")
                             .To<RuntimeChecks()>(XlaOutfeed::Handler())
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void PopulateXlaXfeedCall(xla::runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.cpu.infeed", &xla::cpu::Infeed);
  registry.Register("xla.cpu.outfeed", &xla::cpu::Outfeed);
}

}  // namespace cpu
}  // namespace xla
