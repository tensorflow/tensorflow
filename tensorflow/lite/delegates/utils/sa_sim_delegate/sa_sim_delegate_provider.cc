#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/sa_sim_delegate/sa_sim_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class SASimDelegateProvider : public DelegateProvider {
 public:
  SASimDelegateProvider() {
    default_params_.AddParam("use_sa_sim_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "SASimDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(SASimDelegateProvider);

std::vector<Flag> SASimDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_sa_sim_delegate", params,
                                              "use the sasim delegate.")};
  return flags;
}

void SASimDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_sa_sim_delegate", "Use sasim test delegate",
                 verbose);
}

TfLiteDelegatePtr SASimDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_sa_sim_delegate")) {
    auto default_options = TfLiteSASimDelegateOptionsDefault();
    return TfLiteSASimDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
SASimDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_sa_sim_delegate"));
}
}  // namespace tools
}  // namespace tflite
