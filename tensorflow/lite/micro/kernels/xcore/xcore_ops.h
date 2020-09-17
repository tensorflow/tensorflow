#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr const char* FullyConnected_8_OpCode = "XC_fc";

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
TfLiteRegistration* Register_ArgMax_16();
TfLiteRegistration* Register_MaxPool2D();
TfLiteRegistration* Register_AvgPool2D();
TfLiteRegistration* Register_AvgPool2D_Global();
TfLiteRegistration* Register_Lookup_8();

// legacy operators
TfLiteRegistration* Register_FullyConnected_16();
TfLiteRegistration* Register_Requantize_16_to_8();

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_H_
