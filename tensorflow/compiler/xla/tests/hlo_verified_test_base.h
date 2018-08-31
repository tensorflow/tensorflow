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

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

// A base class for HLO tests that stores a default HloModule, and automatically
// performs verification on that module on tear-down.
class HloVerifiedTestBase : public HloTestBase {
 protected:
  explicit HloVerifiedTestBase(bool layout_sensitive = false,
                               bool allow_mixed_precision = false);
  ~HloVerifiedTestBase() override;

  // Constructs a default shape verifier.
  std::unique_ptr<ShapeVerifier> MakeShapeVerifier();

  // Performs verification on the default HloModule returned by module().
  // Automatically called by the testing framework for each test.
  //
  // REQUIRED: subclasses that override TearDown() must call this explicitly.
  void TearDown() override;

  // Returns the default HloModule, lazily creating it if necessary via
  // HloTestBase::CreateNewModule().
  HloModule& module();
  void ParseAndVerifyModule(absl::string_view hlo_text,
                            const HloModuleConfig& config = HloModuleConfig());

  // Creates a new module for a test, and stores it in modules_ so it can be
  // verified. Intentionally hides HloTestBase::CreateNewModule, to prevent
  // creation of unverified modules.
  HloModule* CreateNewModule(const string& name = TestName());

 private:
  void VerifyModule(HloModule* module);

  // It is confusing to store modules created by module() and CreateNewModule()
  // in different fields, but it allows us to migrate tests to
  // HloVerifiedTestBase more easily, so it's a win because we can verify more
  // modules. See b/80488902.
  //
  // Lazily populated. Access via module().
  std::unique_ptr<HloModule> module_;
  // Populated by calls to CreateNewModule.
  std::vector<std::unique_ptr<HloModule>> modules_;

  bool tear_down_called_ = false;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_HLO_VERIFIED_TEST_BASE_H_
