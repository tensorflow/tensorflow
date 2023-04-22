#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/tconv_fpga_delegate/tconv_fpga_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class TCONVFPGADelegateProvider : public DelegateProvider {
 public:
  TCONVFPGADelegateProvider() {
    default_params_.AddParam("use_tconv_fpga_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "TCONVFPGADelegate"; }
};
REGISTER_DELEGATE_PROVIDER(TCONVFPGADelegateProvider);

std::vector<Flag> TCONVFPGADelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_tconv_fpga_delegate", params,
                                              "use the tconvfpga delegate.")};
  return flags;
}

void TCONVFPGADelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_tconv_fpga_delegate", "Use tconvfpga test delegate",
                 verbose);
}

TfLiteDelegatePtr TCONVFPGADelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_tconv_fpga_delegate")) {
    auto default_options = TfLiteTCONVFPGADelegateOptionsDefault();
    return TfLiteTCONVFPGADelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
TCONVFPGADelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_tconv_fpga_delegate"));
}
}  // namespace tools
}  // namespace tflite
