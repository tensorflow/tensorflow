/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace tools {

TfLiteDelegate* CreateDummyDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  DummyDelegateOptions options = TfLiteDummyDelegateOptionsDefault();

  // Parse key-values options to DummyDelegateOptions by mimicking them as
  // command-line flags.
  std::unique_ptr<const char*> argv =
      std::unique_ptr<const char*>(new const char*[num_options + 1]);
  constexpr char kDummyDelegateParsing[] = "dummy_delegate_parsing";
  argv.get()[0] = kDummyDelegateParsing;

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv.get()[i + 1] = option_args.rbegin()->c_str();
  }

  constexpr char kAllowedBuiltinOp[] = "allowed_builtin_code";
  constexpr char kReportErrorDuingInit[] = "error_during_init";
  constexpr char kReportErrorDuingPrepare[] = "error_during_prepare";
  constexpr char kReportErrorDuingInvoke[] = "error_during_invoke";

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kAllowedBuiltinOp, &options.allowed_builtin_code,
                               "Allowed builtin code."),
      tflite::Flag::CreateFlag(kReportErrorDuingInit,
                               &options.error_during_init,
                               "Report error during init."),
      tflite::Flag::CreateFlag(kReportErrorDuingPrepare,
                               &options.error_during_prepare,
                               "Report error during prepare."),
      tflite::Flag::CreateFlag(kReportErrorDuingInvoke,
                               &options.error_during_invoke,
                               "Report error during invoke."),
  };

  int argc = num_options + 1;
  if (!tflite::Flags::Parse(&argc, argv.get(), flag_list)) {
    return nullptr;
  }

  TFLITE_LOG(INFO) << "Dummy delegate: allowed_builtin_code set to "
                   << options.allowed_builtin_code << ".";
  TFLITE_LOG(INFO) << "Dummy delegate: error_during_init set to "
                   << options.error_during_init << ".";
  TFLITE_LOG(INFO) << "Dummy delegate: error_during_prepare set to "
                   << options.error_during_prepare << ".";
  TFLITE_LOG(INFO) << "Dummy delegate: error_during_invoke set to "
                   << options.error_during_invoke << ".";

  return TfLiteDummyDelegateCreate(&options);
}

}  // namespace tools
}  // namespace tflite

extern "C" {

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::tools::CreateDummyDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteDummyDelegateDelete(delegate);
}

}  // extern "C"
