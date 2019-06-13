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
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Eager op rewrites should inherit from this class and
// implement the Run method.
class EagerOpRewrite {
 public:
  virtual ~EagerOpRewrite() {}

  // To be implemnted by an Eager op rewrite pass.
  virtual Status Run(EagerOperation* orig_op,
                     std::unique_ptr<tensorflow::EagerOperation>& out_op) = 0;

  // Sets the name of the Eager op rewrite.
  void set_name(const string& name) { name_ = name; }

  // Returns the name of the Eager op rewrite.
  string name() const { return name_; }

 private:
  string name_;
};

class EagerOpRewriteRegistry {
 public:
  // Phases at which the Eager op rewrite pass should run.
  // For now we only added PRE_EXECUTION. Expand as needed.
  enum Phase {
    PRE_EXECUTION  // right before executing an eager op
  };

  // Add a rewrite pass to the registry.
  // Only one rewrite pass is allowed per phase.
  void Register(Phase phase, std::unique_ptr<EagerOpRewrite> pass);

  // Run the rewrite pass registered for a given phase.
  Status RunRewrite(Phase phase, EagerOperation* orig_op,
                    std::unique_ptr<tensorflow::EagerOperation>& out_op);

  // Returns the global registry of rewrite passes.
  static EagerOpRewriteRegistry* Global();

 private:
  // Holds all the registered Eager op rewrites.
  std::map<Phase, std::unique_ptr<EagerOpRewrite>> rewrites_;
};

namespace eager_rewrite_registration {

// This class is used to register a new Eager Op rewrite.
class EagerRewriteRegistration {
 public:
  EagerRewriteRegistration(EagerOpRewriteRegistry::Phase phase,
                           std::unique_ptr<EagerOpRewrite> pass,
                           string rewrite_pass_name) {
    pass->set_name(rewrite_pass_name);
    EagerOpRewriteRegistry::Global()->Register(phase, std::move(pass));
  }
};

}  // namespace eager_rewrite_registration

#define REGISTER_REWRITE(phase, rewrite) \
  REGISTER_REWRITE_UNIQ(__COUNTER__, phase, rewrite)

#define REGISTER_REWRITE_UNIQ(ctr, phase, rewrite)                          \
  static ::tensorflow::eager_rewrite_registration::EagerRewriteRegistration \
      register_rewrite_##ctr(                                               \
          phase,                                                            \
          ::std::unique_ptr<::tensorflow::EagerOpRewrite>(new rewrite()),   \
          #rewrite)

}  // namespace tensorflow
#endif
