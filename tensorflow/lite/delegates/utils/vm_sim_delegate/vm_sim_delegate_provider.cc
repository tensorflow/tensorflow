#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/vm_sim_delegate/vm_sim_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class VMSimDelegateProvider : public DelegateProvider {
 public:
  VMSimDelegateProvider() {
    default_params_.AddParam("use_vm_sim_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "VMSimDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(VMSimDelegateProvider);

std::vector<Flag> VMSimDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_vm_sim_delegate", params,
                                              "use the vmsim delegate.")};
  return flags;
}

void VMSimDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_vm_sim_delegate", "Use vmsim test delegate",
                 verbose);
}

TfLiteDelegatePtr VMSimDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_vm_sim_delegate")) {
    auto default_options = TfLiteVMSimDelegateOptionsDefault();
    return TfLiteVMSimDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
VMSimDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_vm_sim_delegate"));
}
}  // namespace tools
}  // namespace tflite
