#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/toy_accelerator/sim_delegate/toy_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class ToyDelegateProvider : public DelegateProvider {
 public:
  ToyDelegateProvider() {
    default_params_.AddParam("use_toy_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "ToyDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(ToyDelegateProvider);

std::vector<Flag> ToyDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_toy_delegate", params,
                                              "use the toy delegate.")};
  return flags;
}

void ToyDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_toy_delegate", "Use toy test delegate",
                 verbose);
}

TfLiteDelegatePtr ToyDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_toy_delegate")) {
    auto default_options = TfLiteToyDelegateOptionsDefault();
    return TfLiteToyDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
ToyDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_toy_delegate"));
}
}  // namespace tools
}  // namespace tflite
