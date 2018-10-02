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

#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

Status VerifiedHloModule::Verify() {
  if (computation_count() == 0) {
    // The computation was never built. Nothing to verify.
    return Status::OK();
  }
  return verifier_.Run(this).status();
}

void VerifiedHloModule::VerifyOrAddFailure(const string& message) {
  Status status = Verify();
  if (!status.ok()) {
    ADD_FAILURE() << "HloVerifier failed on module " << name()
                  << (message.empty() ? "" : absl::StrCat(" (", message, ")"))
                  << ": " << status;
  }
}

HloVerifiedTestBase::HloVerifiedTestBase(bool layout_sensitive,
                                         bool allow_mixed_precision)
    : HloTestBase(
          /*verifier_layout_sensitive=*/layout_sensitive,
          /*allow_mixed_precision_in_hlo_verifier=*/allow_mixed_precision),
      verifier_layout_sensitive_(layout_sensitive),
      allow_mixed_precision_in_hlo_verifier_(allow_mixed_precision) {}

HloModule& HloVerifiedTestBase::module() {
  if (!module_) {
    module_ = CreateNewVerifiedModule(TestName());
  }
  return *module_;
}

HloModule* HloVerifiedTestBase::CreateNewModule(const string& name) {
  modules_.emplace_back(CreateNewVerifiedModule(name));
  return modules_.back().get();
}

void HloVerifiedTestBase::ParseAndVerifyModule(absl::string_view hlo_text,
                                               const HloModuleConfig& config) {
  CHECK(!module_) << "Called ParseModule when test already has a module.";
  module_ = CreateNewVerifiedModule(TestName());
  TF_CHECK_OK(ParseHloString(hlo_text, module_.get()));
  module_->VerifyOrAddFailure("after parsing");
}

StatusOr<std::unique_ptr<VerifiedHloModule>>
HloVerifiedTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, const HloModuleConfig& config) {
  auto module = CreateNewVerifiedModule(TestName());
  TF_RETURN_IF_ERROR(ParseHloString(hlo_text, module.get()));
  TF_RETURN_IF_ERROR(module->Verify());
  return std::move(module);
}

std::unique_ptr<VerifiedHloModule> HloVerifiedTestBase::CreateNewVerifiedModule(
    const string& name) {
  return absl::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(), verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_);
}

}  // namespace xla
