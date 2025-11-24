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

#include "xla/hlo/transforms/shape_canonicalizer.h"

#include <cstddef>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_pool.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {

class ShapeCanonicalizerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ShapeCanonicalizerVisitor(ShapePool* shape_pool)
      : shape_pool_(shape_pool) {}

  absl::Status DefaultAction(HloInstruction* hlo) final {
    MarkAsMaybeChanged(hlo->Canonicalize(shape_pool_));
    return absl::OkStatus();
  }

 private:
  ShapePool* shape_pool_;
};

}  // namespace

ShapeCanonicalizer::ShapeCanonicalizer(ShapePool* shape_pool)
    : shape_pool_(shape_pool) {}

absl::StatusOr<bool> ShapeCanonicalizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Every time we canonicalize shapes in a module, we garbage collect expired
  // shapes from the pool.
  size_t num_erased = shape_pool_->GarbageCollect();
  VLOG(3) << "Garbage collected " << num_erased << " expired shapes";

  ShapeCanonicalizerVisitor visitor(shape_pool_);
  TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&visitor));
  return visitor.changed();
}

}  // namespace xla
