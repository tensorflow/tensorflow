#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_sa_delegate/secda_sa_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class SecdaSADelegateProvider : public DelegateProvider {
 public:
  SecdaSADelegateProvider() {
    default_params_.AddParam("use_secda_sa_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "SecdaSADelegate"; }
};
REGISTER_DELEGATE_PROVIDER(SecdaSADelegateProvider);

std::vector<Flag> SecdaSADelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_secda_sa_delegate", params,
                                              "use the secda_sa delegate.")};
  return flags;
}

void SecdaSADelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_secda_sa_delegate", "Use secda_sa test delegate",
                 verbose);
}

TfLiteDelegatePtr SecdaSADelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_secda_sa_delegate")) {
    auto default_options = TfLiteSecdaSADelegateOptionsDefault();
    return TfLiteSecdaSADelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
SecdaSADelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_secda_sa_delegate"));
}
}  // namespace tools
}  // namespace tflite
