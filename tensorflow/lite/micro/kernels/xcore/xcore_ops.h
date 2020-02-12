#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "flatbuffers/flexbuffers.h"

extern "C" {
    #include "nn_operator.h"
    #include "nn_types.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

TfLiteRegistration* Register_Conv2D_SIDO();
TfLiteRegistration* Register_Conv2D_DIDO();
TfLiteRegistration* Register_FullyConnected_16();
TfLiteRegistration* Register_ArgMax_16();
TfLiteRegistration* Register_MaxPool();
TfLiteRegistration* Register_AvgPool();
TfLiteRegistration* Register_Requantize_16_to_8();

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_H_
