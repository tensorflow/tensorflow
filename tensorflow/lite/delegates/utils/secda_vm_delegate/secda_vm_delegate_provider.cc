#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_vm_delegate/secda_vm_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class SecdaVMDelegateProvider : public DelegateProvider {
 public:
  SecdaVMDelegateProvider() {
    default_params_.AddParam("use_secda_vm_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "SecdaVMDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(SecdaVMDelegateProvider);

std::vector<Flag> SecdaVMDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_secda_vm_delegate", params,
                                              "use the secda_vm delegate.")};
  return flags;
}

void SecdaVMDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_secda_vm_delegate", "Use secda_vm test delegate",
                 verbose);
}

TfLiteDelegatePtr SecdaVMDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_secda_vm_delegate")) {
    auto default_options = TfLiteSecdaVMDelegateOptionsDefault();
    return TfLiteSecdaVMDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
SecdaVMDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_secda_vm_delegate"));
}
}  // namespace tools
}  // namespace tflite
