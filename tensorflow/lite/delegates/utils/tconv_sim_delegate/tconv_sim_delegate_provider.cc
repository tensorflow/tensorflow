#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/tconv_sim_delegate/tconv_sim_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class TCONVSimDelegateProvider : public DelegateProvider {
 public:
  TCONVSimDelegateProvider() {
    default_params_.AddParam("use_tconv_sim_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "TCONVSimDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(TCONVSimDelegateProvider);

std::vector<Flag> TCONVSimDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_tconv_sim_delegate", params,
                                              "use the tconvsim delegate.")};
  return flags;
}

void TCONVSimDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_tconv_sim_delegate", "Use tconvsim test delegate",
                 verbose);
}

TfLiteDelegatePtr TCONVSimDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_tconv_sim_delegate")) {
    auto default_options = TfLiteTCONVSimDelegateOptionsDefault();
    return TfLiteTCONVSimDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
TCONVSimDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_tconv_sim_delegate"));
}
}  // namespace tools
}  // namespace tflite
