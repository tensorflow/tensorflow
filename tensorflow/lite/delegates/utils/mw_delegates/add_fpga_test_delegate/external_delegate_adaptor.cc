#include <cstddef>
#include <string>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/external/external_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/mw_delegates/add_fpga_test_delegate/add_cpu_test_delegate.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace tools {
    TfLiteDelegate* CreateAddFpgaTestDelegateFromOptions(
        const char* const* options_keys, const char* const* options_values,
        size_t num_options) {
      AddFpgaTestDelegateOptions options = TfLiteAddFpgaTestDelegateOptionsDefault();

      // Parse key-values options to SASimDelegateOptions by mimicking them as
    // command-line flags.
    std::vector<const char*> argv;
    argv.reserve(num_options + 1);
    constexpr char kAddFpgaTestDelegateParsing[] = "add_cpu_test_delegate_parsing";
    argv.push_back(kAddFpgaTestDelegateParsing);

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
    constexpr char kReportErrorDuringInit[] = "error_during_init";
    constexpr char kReportErrorDuringPrepare[] = "error_during_prepare";
    constexpr char kReportErrorDuringInvoke[] = "error_during_invoke";

    std::vector<tflite::Flag> flag_list = {
        tflite::Flag::CreateFlag(kAllowedBuiltinOp, &options.allowed_builtin_code,
                                 "Allowed builtin code."),
        tflite::Flag::CreateFlag(kReportErrorDuringInit,
                                 &options.error_during_init,
                                 "Report error during init."),
        tflite::Flag::CreateFlag(kReportErrorDuringPrepare,
                                 &options.error_during_prepare,
                                 "Report error during prepare."),
        tflite::Flag::CreateFlag(kReportErrorDuringInvoke,
                                 &options.error_during_invoke,
                                 "Report error during invoke."),
    };

    int argc = num_options + 1;
    if (!tflite::Flags::Parse(&argc, argv.data(), flag_list)) {
        return nullptr;
    }

    TFLITE_LOG(INFO) << "Add CPU test delegate: allowed_builtin_code set to "
                     << options.allowed_builtin_code << ".";
    TFLITE_LOG(INFO) << "Add CPU test delegate: error_during_init set to "
                     << options.error_during_init << ".";
    TFLITE_LOG(INFO) << "Add CPU test delegate: error_during_prepare set to "
                     << options.error_during_prepare << ".";
    TFLITE_LOG(INFO) << "Add CPU test delegate: error_during_invoke set to "
                     << options.error_during_invoke << ".";

    return TfLiteAddFpgaTestDelegateCreate(&options);
    }
}  // namespace tools
}  // namespace tflite

extern "C" {

    // Defines two symbols that need to be exported to use the TFLite external
// delegate. See tensorflow/lite/delegates/external for details.
extern TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate*
tflite_plugin_create_delegate(const char* const* options_keys,
                              const char* const* options_values,
                              size_t num_options,
                              void (*report_error)(const char*)) {
  return tflite::tools::CreateAddFpgaTestDelegateFromOptions(
      options_keys, options_values, num_options);
}

TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(
    TfLiteDelegate* delegate) {
  TfLiteAddFpgaTestDelegateDelete(delegate);
}

}  // extern "C"

            