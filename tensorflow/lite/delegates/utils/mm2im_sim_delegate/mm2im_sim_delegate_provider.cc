#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/mm2im_sim_delegate/mm2im_sim_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class MM2IMSimDelegateProvider : public DelegateProvider {
 public:
  MM2IMSimDelegateProvider() {
    default_params_.AddParam("use_mm2im_sim_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "MM2IMSimDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(MM2IMSimDelegateProvider);

std::vector<Flag> MM2IMSimDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_mm2im_sim_delegate", params,
                                              "use the mm2imsim delegate.")};
  return flags;
}

void MM2IMSimDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_mm2im_sim_delegate", "Use mm2imsim test delegate",
                 verbose);
}

TfLiteDelegatePtr MM2IMSimDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_mm2im_sim_delegate")) {
    auto default_options = TfLiteMM2IMSimDelegateOptionsDefault();
    return TfLiteMM2IMSimDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
MM2IMSimDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_mm2im_sim_delegate"));
}
}  // namespace tools
}  // namespace tflite
