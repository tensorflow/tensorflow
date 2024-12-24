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

#include "xla/hlo/transforms/literal_canonicalizer.h"

#include <cstddef>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_pool.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace {

class LiteralCanonicalizerVisitor : public DfsHloRewriteVisitor {
 public:
  LiteralCanonicalizerVisitor(LiteralPool* literal_pool, size_t min_size_bytes)
      : literal_pool_(literal_pool), min_size_bytes_(min_size_bytes) {}

  absl::Status HandleConstant(HloInstruction* hlo) final {
    auto* constant = Cast<HloConstantInstruction>(hlo);
    if (constant->HasLiteral() &&
        constant->literal().size_bytes() >= min_size_bytes_) {
      MarkAsMaybeChanged(constant->Canonicalize(literal_pool_));
    }
    return absl::OkStatus();
  }

 private:
  LiteralPool* literal_pool_;
  size_t min_size_bytes_;
};

}  // namespace

LiteralCanonicalizer::LiteralCanonicalizer(LiteralPool* literal_pool,
                                           size_t min_size_bytes)
    : literal_pool_(literal_pool), min_size_bytes_(min_size_bytes) {}

absl::StatusOr<bool> LiteralCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Every time we canonicalize literals in a module, we garbage collect expired
  // literals from the pool.
  size_t num_erased = literal_pool_->GarbageCollect();
  VLOG(3) << "Garbage collected " << num_erased << " expired literals";

  LiteralCanonicalizerVisitor visitor(literal_pool_, min_size_bytes_);
  TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&visitor));
  return visitor.changed();
}

}  // namespace xla
