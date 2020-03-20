#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {


XcoreOpsResolver::XcoreOpsResolver() {
    AddCustom("XC_argmax_16", Register_ArgMax_16());
    AddCustom("XC_maxpool2d", Register_MaxPool2D());
    AddCustom("XC_avgpool2d", Register_AvgPool2D());
    AddCustom("XC_avgpool2d_global", Register_AvgPool2D_Global());
    AddCustom("XC_fc_deepin_anyout", Register_FullyConnected_16());
    AddCustom("XC_conv2d_shallowin_deepout_relu", Register_Conv2D_SIDO());
    AddCustom("XC_conv2d_deepin_deepout_relu", Register_Conv2D_DIDO());
    AddCustom("XC_conv2d_1x1", Register_Conv2D_1x1());
    AddCustom("XC_conv2d_depthwise", Register_Conv2D_depthwise());
    AddCustom("XC_requantize_16_to_8", Register_Requantize_16_to_8());
    AddCustom("XC_lookup_8", Register_Lookup_8());
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite