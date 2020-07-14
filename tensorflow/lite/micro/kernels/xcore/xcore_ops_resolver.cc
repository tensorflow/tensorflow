#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void add_builtin_ops(MicroMutableOpResolver<num_xcore_ops> *resolver) {
  resolver->AddSoftmax();
  resolver->AddPad();
}

void add_custom_ops(MicroMutableOpResolver<num_xcore_ops> *resolver) {
  resolver->AddCustom("XC_argmax_16", Register_ArgMax_16());
  resolver->AddCustom("XC_maxpool2d", Register_MaxPool2D());
  resolver->AddCustom("XC_avgpool2d", Register_AvgPool2D());
  resolver->AddCustom("XC_avgpool2d_global", Register_AvgPool2D_Global());
  resolver->AddCustom("XC_fc_deepin_anyout", Register_FullyConnected_16());
  resolver->AddCustom("XC_conv2d_shallowin", Register_Conv2D_Shallow());
  resolver->AddCustom("XC_conv2d_deep", Register_Conv2D_Deep());
  resolver->AddCustom("XC_conv2d_1x1", Register_Conv2D_1x1());
  resolver->AddCustom("XC_conv2d_depthwise", Register_Conv2D_Depthwise());
  resolver->AddCustom("XC_requantize_16_to_8", Register_Requantize_16_to_8());
  resolver->AddCustom("XC_lookup_8", Register_Lookup_8());
}

XcoreOpsResolver::XcoreOpsResolver() {
  add_builtin_ops(this);
  add_custom_ops(this);
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite