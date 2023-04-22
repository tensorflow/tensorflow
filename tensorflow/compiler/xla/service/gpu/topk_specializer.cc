/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/topk_specializer.h"

#include <stddef.h>

#include <initializer_list>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/tuple_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

StatusOr<HloInstruction*> SmallBufferOptimization(
    HloCustomCallInstruction* topk) {
  Shape data_shape = topk->operand(0)->shape();
  auto supported_dtypes = {F32, BF16};
  if (!absl::c_linear_search(supported_dtypes, data_shape.element_type())) {
    return InvalidArgument(
        "Invalid Dtype: %s",
        primitive_util::LowercasePrimitiveTypeName(data_shape.element_type()));
  }
  // We only support topk of the shape [x] or [batch, x].
  if (data_shape.dimensions_size() > 2) {
    return InvalidArgument("Invalid input dimensions: %s",
                           data_shape.ToString());
  }
  bool has_batch = data_shape.dimensions_size() == 2;
  constexpr size_t max_k = 16;
  constexpr size_t min_n = 1024;
  // TODO(doak): Remove this bound once we have a topk splitter pass.
  constexpr size_t max_n = 1024 * 1024;
  size_t n = data_shape.dimensions(has_batch ? 1 : 0);
  size_t k = topk->shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0);
  if (k > max_k) {
    return InvalidArgument("k too large (%d), must be <= %d", k, max_k);
  }
  if (n < min_n) {
    return InvalidArgument("Input too small (n=%d, min_n=%d)", n, min_n);
  }
  if (n > max_n) {
    return InvalidArgument("Input too large (n=%d, min_n=%d)", n, max_n);
  }
  if (n % 512) {
    return InvalidArgument("Input must be divisible by 512");
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
  Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "TopK") {
      return OkStatus();
    }
    TF_RET_CHECK(topk->operand_count() == 1);

    if (auto small_topk = SmallBufferOptimization(topk); small_topk.ok()) {
      return ReplaceInstruction(topk, *small_topk);
    } else {
      VLOG(2) << "Small TopK optimization doesn't match: "
              << small_topk.status();
    }

    return OkStatus();
  }
};

}  // namespace

StatusOr<bool> TopkSpecializer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return SpecializeTopkVisitor().RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
