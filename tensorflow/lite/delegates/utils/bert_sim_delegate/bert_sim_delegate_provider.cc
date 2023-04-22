#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/bert_sim_delegate/bert_sim_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class BertSimDelegateProvider : public DelegateProvider {
 public:
  BertSimDelegateProvider() {
    default_params_.AddParam("use_bert_sim_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "BertSimDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(BertSimDelegateProvider);

std::vector<Flag> BertSimDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_bert_sim_delegate", params,
                                              "use the bert_sim delegate.")};
  return flags;
}

void BertSimDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_bert_sim_delegate", "Use bert_sim test delegate",
                 verbose);
}

TfLiteDelegatePtr BertSimDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_bert_sim_delegate")) {
    auto default_options = TfLiteBertSimDelegateOptionsDefault();
    return TfLiteBertSimDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
BertSimDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_bert_sim_delegate"));
}
}  // namespace tools
}  // namespace tflite
