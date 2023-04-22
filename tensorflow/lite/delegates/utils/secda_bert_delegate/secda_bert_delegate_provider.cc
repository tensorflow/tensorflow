#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_bert_delegate/secda_bert_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class SecdaBertDelegateProvider : public DelegateProvider {
 public:
  SecdaBertDelegateProvider() {
    default_params_.AddParam("use_secda_bert_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "SecdaBertDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(SecdaBertDelegateProvider);

std::vector<Flag> SecdaBertDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_secda_bert_delegate", params,
                                              "use the secda_bert delegate.")};
  return flags;
}

void SecdaBertDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_secda_bert_delegate", "Use secda_bert test delegate",
                 verbose);
}

TfLiteDelegatePtr SecdaBertDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_secda_bert_delegate")) {
    auto default_options = TfLiteSecdaBertDelegateOptionsDefault();
    return TfLiteSecdaBertDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
SecdaBertDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_secda_bert_delegate"));
}
}  // namespace tools
}  // namespace tflite
