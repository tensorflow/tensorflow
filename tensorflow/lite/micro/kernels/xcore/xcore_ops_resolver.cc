#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void add_custom_ops(MicroMutableOpResolver *resolver) {
  resolver->AddCustom("XC_argmax_16", Register_ArgMax_16());
  resolver->AddCustom("XC_maxpool2d", Register_MaxPool2D());
  resolver->AddCustom("XC_avgpool2d", Register_AvgPool2D());
  resolver->AddCustom("XC_avgpool2d_global", Register_AvgPool2D_Global());
  resolver->AddCustom("XC_fc_deepin_anyout", Register_FullyConnected_16());
  resolver->AddCustom("XC_conv2d_shallowin_deepout_relu", Register_Conv2D_SIDO());
  resolver->AddCustom("XC_conv2d_deep", Register_Conv2D_Deep());
  resolver->AddCustom("XC_conv2d_1x1", Register_Conv2D_1x1());
  resolver->AddCustom("XC_conv2d_depthwise", Register_Conv2D_depthwise());
  resolver->AddCustom("XC_requantize_16_to_8", Register_Requantize_16_to_8());
  resolver->AddCustom("XC_lookup_8", Register_Lookup_8());
}

XcoreOpsResolver::XcoreOpsResolver() {
  add_custom_ops(this);
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite