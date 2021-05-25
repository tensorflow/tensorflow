/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/eager/eager_op_rewrite_registry.h"

namespace tensorflow {

EagerOpRewriteRegistry* EagerOpRewriteRegistry::Global() {
  static EagerOpRewriteRegistry* global_rewrite_registry =
      new EagerOpRewriteRegistry;
  return global_rewrite_registry;
}

void EagerOpRewriteRegistry::Register(Phase phase, int32 ordinal,
                                      std::unique_ptr<EagerOpRewrite> pass) {
  std::list<int32>::const_iterator it_ordinals = ordinals_[phase].cbegin();
  std::list<std::unique_ptr<EagerOpRewrite>>::const_iterator it_rewrites =
      rewrites_[phase].cbegin();
  for (; it_ordinals != ordinals_[phase].cend(); ++it_ordinals, ++it_rewrites) {
    if (*it_ordinals == ordinal) {
      TF_CHECK_OK(errors::AlreadyExists(
          "Attempting to register Eager Rewriter ", pass->GetDebugInfo().name,
          " for phase ", phase, " using ordinal ", ordinal,
          " already occupied by Rewriter ",
          (*it_rewrites)->GetDebugInfo().name));
    }
    if (*it_ordinals > ordinal) {
      break;
    }
  }
  ordinals_[phase].emplace(it_ordinals, ordinal);
  rewrites_[phase].emplace(it_rewrites, std::move(pass));
}

Status EagerOpRewriteRegistry::RunRewrite(
    Phase phase, EagerOperation* orig_op,
    std::unique_ptr<EagerOperation>* out_op) {
  EagerOperation* pre_op = orig_op;
  std::unique_ptr<EagerOperation>* post_op = out_op;
  for (std::list<std::unique_ptr<EagerOpRewrite>>::const_iterator it_rewrites =
           rewrites_[phase].cbegin();
       it_rewrites != rewrites_[phase].cend(); ++it_rewrites) {
    TF_RETURN_IF_ERROR((*it_rewrites)->Run(pre_op, post_op));
    if (*post_op != nullptr) {
      pre_op = post_op->release();
      out_op->reset(pre_op);
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
