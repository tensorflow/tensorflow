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

#include "xla/backends/gpu/transforms/topk_specializer.h"

#include <stddef.h>

#include <initializer_list>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

absl::StatusOr<HloInstruction*> SmallBufferOptimization(
    HloCustomCallInstruction* topk, bool is_cuda,
    bool xla_gpu_experimental_use_raft_select_k) {
  Shape data_shape = topk->operand(0)->shape();
  auto dtype = data_shape.element_type();
  auto supported_dtypes = {F32, BF16};
  if (!absl::c_linear_search(supported_dtypes, dtype)) {
    return InvalidArgument("Invalid Dtype: %s",
                           primitive_util::LowercasePrimitiveTypeName(dtype));
  }
  // We only support topk of the shape [x] or [batch, x].
  if (data_shape.dimensions().size() > 2) {
    return InvalidArgument("Invalid input dimensions: %s",
                           data_shape.ToString());
  }
  bool has_batch = data_shape.dimensions().size() == 2;
  size_t max_k = 16;  // CustomCall TopK requires k <= 16 and n >= 1024
  size_t min_n = 1024;
  size_t batch = 0;
  if (has_batch) {
    batch = data_shape.dimensions(0);
  }
  size_t n = data_shape.dimensions(has_batch ? 1 : 0);
  size_t k = topk->shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0);
  double ratio = static_cast<double>(k) / n;
  if (ratio >= 0.85) {
    return InvalidArgument(
        "k/n ratio (%f) is too high for TopK. Falling back to sort + slice.",
        ratio);
  }
  if (is_cuda && xla_gpu_experimental_use_raft_select_k) {
    // The heuristic for deciding when to use Raft select_k versus Sort + Slice
    // was developed as part of the initial research in b/409009349
    if (dtype == F32) {
      min_n = 1;
      max_k = 128;
      if (batch >= 64 && n >= 16384) {
        max_k = 256;
      }
    } else if (dtype == BF16) {
      min_n = 1;
      max_k = 128;
      if (batch >= 16 && n >= 65536) {
        max_k = 256;
      }
      if (batch >= 64 && batch <= 128 && n >= 8192 && n <= 32768) {
        max_k = 64;
      }
    }
  }

  if (k > max_k) {
    return InvalidArgument("k too large (%d), must be <= %d", k, max_k);
  }
  if (n < min_n) {
    return InvalidArgument("Input too small (n=%d, min_n=%d)", n, min_n);
  }
  HloComputation* comp = topk->parent();
  HloInstruction* new_topk =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          topk->shape(), topk->operands(),
          // We don't need the original to_apply, but keeping it around allows
          // us to round-trip this CustomCall on tests.
          topk->to_apply(), "__gpu$TopK",
          /*opaque=*/"", CustomCallApiVersion::API_VERSION_TYPED_FFI));
  return TupleUtil::ExtractPrefix(new_topk, 2);
}

class SpecializeTopkVisitor : public DfsHloRewriteVisitor {
 public:
  explicit SpecializeTopkVisitor(se::GpuComputeCapability compute_capability)
      : compute_capability_(std::move(compute_capability)) {}

  absl::Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "TopK") {
      return absl::OkStatus();
    }
    TF_RET_CHECK(topk->operand_count() == 1);
    bool is_cuda = compute_capability_.IsCuda();

    if (auto small_topk = SmallBufferOptimization(
            topk, is_cuda,
            inst->GetModule()
                ->config()
                .debug_options()
                .xla_gpu_experimental_use_raft_select_k());
        small_topk.ok()) {
      return ReplaceInstruction(topk, *small_topk);
    } else {  // NOLINT(readability-else-after-return)
      VLOG(2) << "Small TopK optimization doesn't match: "
              << small_topk.status();
    }

    return absl::OkStatus();
  }

 private:
  se::GpuComputeCapability compute_capability_;
};

}  // namespace

absl::StatusOr<bool> TopkSpecializer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return SpecializeTopkVisitor(compute_capability_)
      .RunOnModule(module, execution_threads);
}

}  // namespace xla::gpu
