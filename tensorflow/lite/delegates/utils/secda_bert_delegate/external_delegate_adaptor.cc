#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/bert_delegate/bert_delegate.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace tools {

TfLiteDelegate* CreateBertDelegateFromOptions(char** options_keys,
                                               char** options_values,
                                               size_t num_options) {
  BertDelegateOptions options = TfLiteBertDelegateOptionsDefault();

  // Parse key-values options to BertDelegateOptions by mimicking them as
  // command-line flags.
  std::vector<const char*> argv;
  argv.reserve(num_options + 1);
  constexpr char kBertDelegateParsing[] = "bert_delegate_parsing";
  argv.push_back(kBertDelegateParsing);

  std::vector<std::string> option_args;
  option_args.reserve(num_options);
  for (int i = 0; i < num_options; ++i) {
    option_args.emplace_back("--");
    option_args.rbegin()->append(options_keys[i]);
    option_args.rbegin()->push_back('=');
    option_args.rbegin()->append(options_values[i]);
    argv.push_back(option_args.rbegin()->c_str());
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
  if (!tflite::Flags::Parse(&argc, argv.data(), flag_list)) {
    return nullptr;
  }

  TFLITE_LOG(INFO) << "Bert delegate: allowed_builtin_code set to "
                   << options.allowed_builtin_code << ".";
  TFLITE_LOG(INFO) << "Bert delegate: error_during_init set to "
                   << options.error_during_init << ".";
  TFLITE_LOG(INFO) << "Bert delegate: error_during_prepare set to "
                   << options.error_during_prepare << ".";
  TFLITE_LOG(INFO) << "Bert delegate: error_during_invoke set to "
                   << options.error_during_invoke << ".";

  return TfLiteBertDelegateCreate(&options);
}

}  // namespace tools
}  // namespace tflite

extern "C" {

// Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys, char** options_values, size_t num_options,
    void (*report_error)(const char*)) {
  return tflite::tools::CreateBertDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteBertDelegateDelete(delegate);
}

}  // extern "C"
