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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OP_REWRITE_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OP_REWRITE_REGISTRY_H_

#include <map>
#include <vector>

#include "tensorflow/core/common_runtime/eager/eager_operation.h"

namespace tensorflow {

// Eager op rewrites should inherit from this class and
// implement the Run method.
class EagerOpRewrite {
 public:
  EagerOpRewrite(string name, string file, string line) {
    debug_info_.name = name;
    debug_info_.file = file;
    debug_info_.line = line;
  }

  virtual ~EagerOpRewrite() {}

  // To be implemented by an Eager op rewrite pass.
  virtual Status Run(EagerOperation* orig_op,
                     std::unique_ptr<tensorflow::EagerOperation>* out_op) = 0;

  // Holds information about the rewrite registration.
  struct DebugInfo {
    string name, file, line;
  };

  // Returns information about the registered Eager op rewrite.
  DebugInfo GetDebugInfo() const { return debug_info_; }

 private:
  DebugInfo debug_info_;
};

class EagerOpRewriteRegistry {
 public:
  // Phases at which the Eager op rewrite pass should run.
  // For now we only added PRE_EXECUTION. Expand as needed.
  enum Phase {
    PRE_EXECUTION = 0,  // right before executing an eager op
    POST_PLACEMENT = 1  // after device placement
  };

  // Add a rewrite pass to the registry.
  // Only one rewrite pass is allowed per phase.
  void Register(Phase phase, std::unique_ptr<EagerOpRewrite> pass);

  // Run the rewrite pass registered for a given phase.
  Status RunRewrite(Phase phase, EagerOperation* orig_op,
                    std::unique_ptr<tensorflow::EagerOperation>* out_op);

  // Returns the global registry of rewrite passes.
  static EagerOpRewriteRegistry* Global();

 private:
  static constexpr int32 kNumPhases = 2;
  // Holds all the registered Eager op rewrites.
  std::array<std::unique_ptr<EagerOpRewrite>, kNumPhases> rewrites_;
};

namespace eager_rewrite_registration {

// This class is used to register a new Eager Op rewrite.
class EagerRewriteRegistration {
 public:
  EagerRewriteRegistration(EagerOpRewriteRegistry::Phase phase,
                           std::unique_ptr<EagerOpRewrite> pass) {
    EagerOpRewriteRegistry::Global()->Register(phase, std::move(pass));
  }
};

}  // namespace eager_rewrite_registration

#define REGISTER_REWRITE(phase, rewrite) \
  REGISTER_REWRITE_UNIQ_HELPER(__COUNTER__, __FILE__, __LINE__, phase, rewrite)

#define REGISTER_REWRITE_UNIQ_HELPER(ctr, file, line, phase, rewrite) \
  REGISTER_REWRITE_UNIQ(ctr, file, line, phase, rewrite)

#define REGISTER_REWRITE_UNIQ(ctr, file, line, phase, rewrite)                \
  static ::tensorflow::eager_rewrite_registration::EagerRewriteRegistration   \
      register_rewrite_##ctr(phase,                                           \
                             ::std::unique_ptr<::tensorflow::EagerOpRewrite>( \
                                 new rewrite(#rewrite, file, #line)))

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_OP_REWRITE_REGISTRY_H_
