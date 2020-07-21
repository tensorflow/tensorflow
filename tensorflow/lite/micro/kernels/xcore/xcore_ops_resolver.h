#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr int num_xcore_ops = 13;

class XCoreOpsResolver : public MicroMutableOpResolver<num_xcore_ops> {
 public:
  XCoreOpsResolver() {}
  void AddXC();

  TfLiteStatus AddXC_ArgMax() {
    return AddCustom("XC_argmax_16", Register_ArgMax_16());
  }

  TfLiteStatus AddXC_MaxPool2D() {
    return AddCustom("XC_maxpool2d", Register_MaxPool2D());
  }

  TfLiteStatus AddXC_AvgPool2D() {
    return AddCustom("XC_avgpool2d", Register_AvgPool2D());
  }

  TfLiteStatus AddXC_AvgPoolGlobal() {
    return AddCustom("XC_avgpool2d_global", Register_AvgPool2D_Global());
  }

  TfLiteStatus AddXC_FullyConnected() {
    return AddCustom("XC_fc_deepin_anyout", Register_FullyConnected_16());
  }

  TfLiteStatus AddXC_Conv2D_Shallow() {
    return AddCustom("XC_conv2d_shallowin", Register_Conv2D_Shallow());
  }

  TfLiteStatus AddXC_Conv2D_Deep() {
    return AddCustom("XC_conv2d_deep", Register_Conv2D_Deep());
  }

  TfLiteStatus AddXC_Conv2D_1x1() {
    return AddCustom("XC_conv2d_1x1", Register_Conv2D_1x1());
  }

  TfLiteStatus AddXC_Conv2D_Deepthwise() {
    return AddCustom("XC_conv2d_depthwise", Register_Conv2D_Depthwise());
  }

  TfLiteStatus AddXC_Requantize_16_to_8() {
    return AddCustom("XC_requantize_16_to_8", Register_Requantize_16_to_8());
  }

  TfLiteStatus AddXC_Lookup8() {
    return AddCustom("XC_lookup_8", Register_Lookup_8());
  }

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_
