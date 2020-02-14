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

#include "tensorflow/compiler/xla/tools/run_hlo_module.h"

#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/testing.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/compiler/xla/tools/prepare_reference_module.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"

namespace se = ::stream_executor;

namespace xla {
namespace {

Literal ExecuteOnPlatform(std::unique_ptr<HloModule> module,
                          absl::Span<const Literal> args,
                          se::Platform* platform, bool run_hlo_passes) {
  HloRunner runner(platform);

  TF_QCHECK_OK(VerifyHloModule(module.get(), /*layout_sensitive=*/false,
                               /*allow_mixed_precision=*/true))
      << " (on " << platform->Name() << ")";

  std::cerr << "Running HLO module on platform " << platform->Name() << "...\n";
  XLA_VLOG_LINES(1, module->ToString());
  const auto start = std::chrono::high_resolution_clock::now();
  auto result_status = runner.Execute(std::move(module), args, run_hlo_passes);
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << "... compiled and ran in " << diff.count() << "s.\n";

  TF_QCHECK_OK(result_status.status())
      << "Failed to execute on " << platform->Name() << "\n";

  return result_status.ConsumeValueOrDie();
}
}  // namespace

::testing::AssertionResult RunAndCompare(
    const std::string& hlo_filename, const std::string& test_platform_name,
    const std::string& reference_platform_name, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    std::function<Status(const HloModule&,
                         const ::stream_executor::Platform::Id&, HloModule*)>
        reference_module_modifier_hook) {
  se::Platform* test_platform =
      xla::PlatformUtil::GetPlatform(test_platform_name).ValueOrDie();
  se::Platform* reference_platform =
      reference_platform_name.empty()
          ? nullptr
          : xla::PlatformUtil::GetPlatform(reference_platform_name)
                .ValueOrDie();
  auto config_modifier = [](HloModuleConfig* config) { config->set_seed(42); };

  std::unique_ptr<HloModule> test_module =
      LoadModuleFromFile(hlo_filename, hlo_module_loader_details::Config(),
                         options.input_format, config_modifier)
          .ValueOrDie();
  const HloModuleProto test_module_proto = test_module->ToProto();

  std::vector<Literal> args = MakeFakeArguments(test_module.get(), engine,
                                                options.use_large_float_range)
                                  .ConsumeValueOrDie();

  if (options.print_literals) {
    for (int i = 0; i < args.size(); ++i) {
      std::cout << "\n** Argument " << i << " **\n"
                << args[i].ToString() << "\n";
    }
  }

  std::unique_ptr<HloModule> reference_module;
  if (reference_platform != nullptr) {
    // PrepareReferenceModule needs to know the *test* platform, in order to
    // properly match the test platform's numerics.
    reference_module =
        PrepareReferenceModule(*test_module, test_platform->id(),
                               config_modifier, reference_module_modifier_hook)
            .ConsumeValueOrDie();
  }

  Literal test_result = ExecuteOnPlatform(
      std::move(test_module), args, test_platform, options.run_test_hlo_passes);
  if (options.print_literals) {
    std::cout << "\n** Result on test platform " << test_platform->Name()
              << " **\n"
              << test_result.ToString() << "\n";
  }

  if (reference_module == nullptr) {
    std::cerr << "Skipping reference platform\n";
    return ::testing::AssertionSuccess();
  }

  Literal reference_result =
      ExecuteOnPlatform(std::move(reference_module), args, reference_platform,
                        options.run_reference_hlo_passes);

  if (options.print_literals) {
    std::cout << "\n** Result on reference platform "
              << reference_platform->Name() << " **\n"
              << reference_result.ToString() << "\n";
  }
  ErrorSpec error_spec(static_cast<float>(options.abs_error_bound),
                       static_cast<float>(options.rel_error_bound));
  return LiteralTestUtil::Near(/*expected=*/reference_result,
                               /*actual=*/test_result,
                               /*error_spec=*/error_spec,
                               /*detailed_message=*/true);
}

}  // namespace xla
