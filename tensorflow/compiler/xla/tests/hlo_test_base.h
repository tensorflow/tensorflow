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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

// A base class for tests which build and/or run HLO code. The class includes
// support for running an HLO module on two platforms and compare the results.
// This is a lower level of abstraction than using the client interface and
// enables, for one, explicitly building a graph of HLO instructions to run.
//
// This can also be used to write text/file-based test cases. Note that the test
// target is responsible for linking the needed backends. A covenient way to do
// this is to make it an xla_test: it will generate test targets linking with
// the respective backends, which will be used as the test backend; the
// interpreter backend is already linked with hlo_test_base so it will be the
// default reference backend. For example, if you want to compare both cpu vs.
// interpreter, and gpu vs. interpreter, you can:
//
//  xla_test (
//    name = "sample_text_test",
//    srcs = ["sample_text_test.cc"],
//    backends = [
//      "cpu",
//      "gpu",
//    ],
//    deps = [
//      "//third_party/tensorflow/compiler/xla/tests:hlo_test_base",
//      ...
//    ],
//  )
//
// For a more detailed example, see "../tests/sample_text_test.cc".
class HloTestBase : public ::testing::Test {
 protected:
  // This uses the interpreter backend as the reference backend and
  // automatically finds another supported backend as the test backend. If the
  // interpreter is the only supported backend, it will be both the test backend
  // and the reference backend.
  HloTestBase();

  // If your test doesn't use interpreter as the reference backend, you can use
  // this constructor. Note that your test target is responsible for linking in
  // both needed backends.
  HloTestBase(::perftools::gputools::Platform* test_platform,
              ::perftools::gputools::Platform* reference_platform);

  ~HloTestBase() override {}

  // Creates a new HLO module for a test. The module created will have
  // TestName() for its name; it will also automatically populate its debug
  // options from command-line flags. If you want a fresh HloModule object and
  // then add HloComputations to it, it's recommended to use this method in your
  // tests.
  static std::unique_ptr<HloModule> CreateNewModule();

  // Populates debug options from command-line flags and adjusts the options for
  // testing. It is recommended to use this when you need to pass in
  // DebugOptions, e.g. when creating a module from a string or a file.
  static DebugOptions GetDebugOptionsForTest();

  // Executes the given module and return the result as a Literal.
  StatusOr<std::unique_ptr<Literal>> Execute(
      std::unique_ptr<HloModule> module,
      tensorflow::gtl::ArraySlice<Literal*> arguments);

  std::unique_ptr<Literal> ExecuteAndTransfer(
      std::unique_ptr<HloModule> module,
      tensorflow::gtl::ArraySlice<Literal*> arguments);

  // Executes the given hlo module on two backends and compares results.
  //
  // 'arguments': the input of the hlo module. The LiteralPtr type accepts
  // Literal* or std::unique_ptr<Literal>.
  //
  // 'error': if has value, expects the results to be near (within the error
  // bound). Otherwise, expects the results to be equal.
  //
  // 'reference_preprocessor': the module should be ready to run on the test
  // backend, but it might need to be tailored so that it is able to run on the
  // reference backend. Note that the program shape of the module must not be
  // modified.
  template <typename LiteralPtr>
  ::testing::AssertionResult RunAndCompare(
      std::unique_ptr<HloModule> module,
      const tensorflow::gtl::ArraySlice<LiteralPtr> arguments,
      const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  template <typename LiteralPtr>
  ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module,
      const tensorflow::gtl::ArraySlice<LiteralPtr> arguments,
      const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;

  // Executes an hlo module with fake inputs and compares the results.
  ::testing::AssertionResult RunAndCompare(
      std::unique_ptr<HloModule> module,
      const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;

  // Same as above, except that the module will be executed without Hlo
  // optimization.
  ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module,
      const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;

  // Convenient wrappers for executing and comparing an hlo module with fake
  // input. Module can be passed in directly, or parsed from an hlo_string,
  // or loaded from a file.
  ::testing::AssertionResult RunAndCompare(
      const tensorflow::StringPiece hlo_string,
      const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;
  ::testing::AssertionResult RunAndCompareFromFile(
      const string& filename, const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;
  ::testing::AssertionResult RunAndCompareNoHloPasses(
      const tensorflow::StringPiece hlo_string,
      const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;
  ::testing::AssertionResult RunAndCompareNoHloPassesFromFile(
      const string& filename, const tensorflow::gtl::optional<ErrorSpec>& error,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr)
      TF_MUST_USE_RESULT;

  // Convenience method to force the layout of a given parameter in a module.
  // The layout of parameter number 'param_no' in the 'module' is set to
  // 'layout'.
  void ForceParameterLayout(HloModule* module, int64 param_no,
                            const Layout& layout) {
    ASSERT_LT(param_no,
              module->mutable_entry_computation_layout()->parameter_count());
    module->mutable_entry_computation_layout()
        ->mutable_parameter_layout(param_no)
        ->ResetLayout(layout);
  }

  // Convenience method to force the layout of the computation result in a
  // module. The result layout of 'module' is set to 'layout'.
  void ForceResultLayout(HloModule* module, const Layout& layout) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->ResetLayout(layout);
  }

  // Convenience method to clear the layout of the computation result in
  // 'module'.
  void ForceClearResultLayout(HloModule* module) {
    module->mutable_entry_computation_layout()
        ->mutable_result_layout()
        ->Clear();
  }

  // Return an HLO verifier constructed for the test backend.
  HloVerifier& verifier() const { return *hlo_verifier_; }

  static string TestName();

  // Returns the backend owned by the test runner.
  Backend& backend();

  HloRunner test_runner_;
  HloRunner reference_runner_;

  std::unique_ptr<HloVerifier> hlo_verifier_;

  ErrorSpec error_spec_{0.0001};

 private:
  // Given the test module, makes a reference module that is ready to run on the
  // reference platform. This assumes that the given module is ready to run on
  // the test platform.
  StatusOr<std::unique_ptr<HloModule>> MakeReferenceModule(
      const HloModule& test_module,
      const std::function<void(HloModule*)>& reference_preprocessor);

  // Runs the module on two platforms with or without running hlo passes and
  // compares the results. Returns whether the results are near or equal. If any
  // error happens before the results are computed, returns the error status.
  template <typename LiteralPtr>
  StatusOr<::testing::AssertionResult> RunAndCompareInternal(
      std::unique_ptr<HloModule> module,
      const tensorflow::gtl::ArraySlice<LiteralPtr> arguments,
      const tensorflow::gtl::optional<ErrorSpec>& error, bool run_hlo_passes,
      const std::function<void(HloModule*)>& reference_preprocessor);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_
