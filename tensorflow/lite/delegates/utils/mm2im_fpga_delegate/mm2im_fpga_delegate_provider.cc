#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/mm2im_fpga_delegate/mm2im_fpga_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class MM2IMFPGADelegateProvider : public DelegateProvider {
 public:
  MM2IMFPGADelegateProvider() {
    default_params_.AddParam("use_mm2im_fpga_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "MM2IMFPGADelegate"; }
};
REGISTER_DELEGATE_PROVIDER(MM2IMFPGADelegateProvider);

std::vector<Flag> MM2IMFPGADelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_mm2im_fpga_delegate", params,
                                              "use the mm2imfpga delegate.")};
  return flags;
}

void MM2IMFPGADelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_mm2im_fpga_delegate", "Use mm2imfpga test delegate",
                 verbose);
}

TfLiteDelegatePtr MM2IMFPGADelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_mm2im_fpga_delegate")) {
    auto default_options = TfLiteMM2IMFPGADelegateOptionsDefault();
    return TfLiteMM2IMFPGADelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
MM2IMFPGADelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_mm2im_fpga_delegate"));
}
}  // namespace tools
}  // namespace tflite
