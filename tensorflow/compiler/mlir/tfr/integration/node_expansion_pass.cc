/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfr/integration/node_expansion_pass.h"

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {

auto* tf_core_op_expansion_node_counter =
    monitoring::Counter<0>::New("/tensorflow/core/op_expansion/node_counter",
                                "The number of nodes being op expanded.");
}  // namespace

namespace tfr {

Status CompositeOpExpansion::Run(EagerOperation* orig_op,
                                 std::unique_ptr<EagerOperation>* out_op) {
  if (!IsEnabled()) return Status::OK();
  // This can be the default cpu device.
  if (orig_op->Device() != kVariantDeviceNull) return Status::OK();
  if (orig_op->is_function()) return Status::OK();

  // TODO(fengliuai): We need a better condition to skip the rewrite. Currently,
  // The rewrite is enabled for all the tf ops and it is a no-op if the tf op
  // isn't a composite op. The following ops are explicitly skipped here because
  // their "no-op" expansion is known to cause problems in some cases.
  static const char* kOpsToSkip[] = {
      "IdentityOp",
      "NoOp",              // b/174596063
      "OptionalHasValue",  // b/173136483
      "OptionalGetValue",  // b/173136483
      "VarHandleOp",       // b/176819198
  };
  for (const char* skip : kOpsToSkip) {
    if (absl::StartsWith(orig_op->op_name(), skip)) return Status::OK();
  }

  tf_core_op_expansion_node_counter->GetCell()->IncrementBy(1);

  LOG_FIRST_N(INFO, 1) << "Run Node Expansion Passes";

  // Get the FunctionDef and insert that into the context
  const NodeDef& ndef = orig_op->MutableAttrs()->BuildNodeDef();
  auto& ctx = orig_op->EagerContext();
  Fprint128 cache_key =
      orig_op->MutableAttrs()->CacheKey(orig_op->DeviceName());
  // Include soft placement policy in cache key since the placement strategy
  // can change and thus affect which kernel is picked.
  auto x = FingerprintCat64(cache_key.high64, cache_key.low64);
  std::string fname =
      absl::StrCat("_expanded_", ndef.name(), "_", std::to_string(x));
  if (!ctx.FindFunctionByName(fname)) {
    TF_ASSIGN_OR_RETURN(auto func, ExpandNode(ndef, fname));
    TF_RETURN_IF_ERROR(ctx.AddFunctionDef(func));
  }

  // Rewrite the out_op to be the call op. This essentially a deep copy of the
  // orig_op, except the op name.
  auto* new_op = new EagerOperation(&ctx);
  TF_RETURN_IF_ERROR(
      new_op->Reset(fname.c_str(), orig_op->DeviceName().c_str()));
  for (auto input : orig_op->GetInputs()) {
    TF_RETURN_IF_ERROR(new_op->AddInput(input));
  }
  new_op->MutableAttrs()->CopyAttributes(orig_op->Attrs());
  out_op->reset(new_op);

  LOG_FIRST_N(INFO, 1)
      << "Finish Node Expansion Passes. Rewrite the op to call function: "
      << fname;

  return Status::OK();
}

REGISTER_REWRITE(EagerOpRewriteRegistry::POST_PLACEMENT, 20000,
                 CompositeOpExpansion);

}  // namespace tfr
}  // namespace tensorflow
