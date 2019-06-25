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

void EagerOpRewriteRegistry::Register(Phase phase,
                                      std::unique_ptr<EagerOpRewrite> pass) {
  if (rewrites_.find(phase) == rewrites_.end()) {
    rewrites_[phase] = std::move(pass);
  } else {
    TF_CHECK_OK(errors::AlreadyExists(
        "An EagerOpRewrite is already registerd for this phase: ",
        pass->name()));
  }
}

Status EagerOpRewriteRegistry::RunRewrite(
    Phase phase, EagerOperation* orig_op,
    std::unique_ptr<tensorflow::EagerOperation>* out_op) {
  auto rewrite = rewrites_.find(phase);
  if (rewrite != rewrites_.end()) {
    Status s = rewrite->second->Run(orig_op, out_op);
    if (!s.ok()) return s;
  }
  return Status::OK();
}

}  // namespace tensorflow
