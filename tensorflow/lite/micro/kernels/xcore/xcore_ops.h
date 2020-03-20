#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

TfLiteRegistration* Register_Conv2D_SIDO();
TfLiteRegistration* Register_Conv2D_DIDO();
TfLiteRegistration* Register_Conv2D_1x1();
TfLiteRegistration* Register_Conv2D_depthwise();
TfLiteRegistration* Register_FullyConnected_16();
TfLiteRegistration* Register_ArgMax_16();
TfLiteRegistration* Register_MaxPool2D();
TfLiteRegistration* Register_AvgPool2D();
TfLiteRegistration* Register_AvgPool2D_Global();
TfLiteRegistration* Register_Requantize_16_to_8();
TfLiteRegistration* Register_Lookup_8();

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_H_
