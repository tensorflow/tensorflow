#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr const char* Lookup_8_OpCode = "XC_lookup_8";
constexpr const char* MaxPool2D_OpCode = "XC_maxpool2d";
constexpr const char* AvgPool2D_OpCode = "XC_avgpool2d";
constexpr const char* AvgPool2D_Global_OpCode = "XC_avgpool2d_global";
constexpr const char* FullyConnected_8_OpCode = "XC_fc";
constexpr const char* Conv2D_Shallow_OpCode = "XC_conv2d_shallowin";
constexpr const char* Conv2D_Deep_OpCode = "XC_conv2d_deep";
constexpr const char* Conv2D_1x1_OpCode = "XC_conv2d_1x1";
constexpr const char* Conv2D_Depthwise_OpCode = "XC_conv2d_depthwise";
constexpr const char* Add_8_OpCode = "XC_add_8";

constexpr const char* Requantize_16_to_8_OpCode = "XC_requantize_16_to_8";
constexpr const char* ArgMax2D_OpCode = "XC_argmax_16";

struct PoolingParams {
  int32_t pool_h;
  int32_t pool_w;
  int32_t stride_h;
  int32_t stride_w;
};

struct Conv2DPadding {
  int8_t top;
  int8_t left;
  int8_t zero_point;
  int8_t unused;
};

struct Conv2DParams {
  int32_t K_h;
  int32_t K_w;
  int32_t stride_h;
  int32_t stride_w;
  Conv2DPadding pad;
};

TfLiteRegistration* Register_Conv2D_Shallow();
TfLiteRegistration* Register_Conv2D_Deep();
TfLiteRegistration* Register_Conv2D_1x1();
TfLiteRegistration* Register_Conv2D_Depthwise();
TfLiteRegistration* Register_FullyConnected_8();
TfLiteRegistration* Register_MaxPool2D();
TfLiteRegistration* Register_AvgPool2D();
TfLiteRegistration* Register_AvgPool2D_Global();
TfLiteRegistration* Register_Lookup_8();
TfLiteRegistration* Register_BSign_8();
TfLiteRegistration* Register_BConv2D_Bin_Out();
TfLiteRegistration* Register_Pad();
TfLiteRegistration* Register_Add_8();

// operators not currently inserted by the XCORE converter
TfLiteRegistration* Register_Requantize_16_to_8();
TfLiteRegistration* Register_ArgMax_16();

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_H_
