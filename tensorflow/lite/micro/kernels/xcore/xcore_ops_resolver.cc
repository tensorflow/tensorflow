#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void XcoreOpsResolver::AddXC() {
  AddSoftmax();
  AddPad();
  AddXC_ArgMax();
  AddXC_MaxPool2D();
  AddXC_AvgPool2D();
  AddXC_AvgPoolGlobal();
  AddXC_FullyConnected();
  AddXC_Conv2D_Shallow();
  AddXC_Conv2D_Deep();
  AddXC_Conv2D_1x1();
  AddXC_Conv2D_Deepthwise();
  AddXC_Requantize_16_to_8();
  AddXC_Lookup8();
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite