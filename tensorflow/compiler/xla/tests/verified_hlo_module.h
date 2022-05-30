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
#ifndef TENSORFLOW_COMPILER_XLA_TESTS_VERIFIED_HLO_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_VERIFIED_HLO_MODULE_H_

#include <functional>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// An HLO module derived class which verifies itself on destruction. This class
// is intended to be used in unit tests. Any verification errors are raised via
// ADD_FAILURE.
class VerifiedHloModule : public HloModule {
 public:
  VerifiedHloModule(const std::string& name, const HloModuleConfig& config,
                    bool verifier_layout_sensitive,
                    bool allow_mixed_precision_in_hlo_verifier,
                    std::function<int64_t(const Shape&)> shape_size_function)
      : HloModule(name, config),
        verifier_(
            verifier_layout_sensitive, allow_mixed_precision_in_hlo_verifier,
            /*instruction_can_change_layout_func=*/{}, shape_size_function) {}

  ~VerifiedHloModule() override { VerifyOrAddFailure("in destructor"); }

  // Given a string in the HloModule::ToString() format, parses the string and
  // builds the VerifiedHloModule in place. Before calling this method, the
  // module must be empty (no computations). Finally verifies the module using
  // HloVerifier and returns the status.
  Status ParseHloStringAndVerifyModule(absl::string_view str);

  // Verifies the module and flags any error with ADD_FAILURE. 'message' is
  // included in the failure message.
  void VerifyOrAddFailure(absl::string_view message);

  // Verifies the module using HloVerifier and returns the status.
  Status Verify();

 private:
  HloVerifier verifier_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_VERIFIED_HLO_MODULE_H_
