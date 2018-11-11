/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_HLO_VERIFIED_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_HLO_VERIFIED_TEST_BASE_H_

#include <functional>
#include <memory>
#include <utility>

#include "absl/base/macros.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

// An HLO module derived class which verifies itself on destruction. This class
// is intended to be used in unit tests. Any verification errors are raised via
// ADD_FAILURE.
class VerifiedHloModule : public HloModule {
 public:
  VerifiedHloModule(const string& name, const HloModuleConfig& config,
                    bool verifier_layout_sensitive,
                    bool allow_mixed_precision_in_hlo_verifier)
      : HloModule(name, config),
        verifier_(verifier_layout_sensitive,
                  allow_mixed_precision_in_hlo_verifier) {}

  ~VerifiedHloModule() override { VerifyOrAddFailure("in destructor"); }

  // Verifies the module using HloVerifier and returns the status.
  Status Verify();

  // Verifies the module and flags any error with ADD_FAILURE. 'message' is
  // included in the failure message.
  void VerifyOrAddFailure(const string& message);

 private:
  HloVerifier verifier_;
};

// A base class for HLO tests that stores a default VerifiedHloModule.
class HloVerifiedTestBase : public HloTestBase {
 protected:
  HloVerifiedTestBase(bool layout_sensitive = false,
                      bool allow_mixed_precision = false);

  // Constructs a default shape verifier.
  std::unique_ptr<ShapeVerifier> MakeShapeVerifier();

  // Parses the given string and returns module as a VerifiedHloModule.
  StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndReturnVerifiedModule(
      absl::string_view hlo_text,
      const HloModuleConfig& config = HloModuleConfig());

  // Creates and returns a verified HLO module with the given name.
  std::unique_ptr<VerifiedHloModule> CreateNewVerifiedModule(
      const string& name = TestName());

  // CreateNewUnverifiedModule creates an *unverified* module, which presumably
  // isn't what you want if you're using HloVerifiedTestBase, so we delete this
  // function to keep you from accidentally calling it.  If you really want it,
  // you can get it by calling HloTestBase::CreateNewUnverifiedModule().
  std::unique_ptr<HloModule> CreateNewUnverifiedModule(
      const string& name = TestName()) = delete;

 private:
  bool verifier_layout_sensitive_;
  bool allow_mixed_precision_in_hlo_verifier_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_HLO_VERIFIED_TEST_BASE_H_
