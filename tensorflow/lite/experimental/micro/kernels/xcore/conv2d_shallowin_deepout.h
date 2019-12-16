#ifndef XCORE_OPS_CONV2D_SHALLOWOUT_DEEPOUT_H_
#define XCORE_OPS_CONV2D_SHALLOWOUT_DEEPOUT_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

extern "C" {
    #include "nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {

TfLiteStatus Conv2DShallowinDeepoutPrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 5);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    return kTfLiteOk;
}

TfLiteStatus Conv2DShallowinDeepoutEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = GetInput(context, node, 0);
    const TfLiteTensor* weights = GetInput(context, node, 1);
    const TfLiteTensor* biases = GetInput(context, node, 2);
    const TfLiteTensor* shift_scale = GetInput(context, node, 3);
    const TfLiteTensor* unpadded_shape = GetInput(context, node, 4);

    int32_t height = input->dims->data[1];
    int32_t width = input->dims->data[2];
    int32_t C_out = unpadded_shape->data.i32[0];
    int32_t K_h = unpadded_shape->data.i32[1];
    int32_t K_w = unpadded_shape->data.i32[2];
    int32_t scales_offset = C_out;

    TfLiteTensor* output = GetOutput(context, node, 0);

    conv2d_shallowin_deepout_relu(
        weights->data.int8,
        (data16_t *) biases->data.i16,
        input->data.int8,
        output->data.int8,
        height,
        width,
        K_h,
        K_w,
        C_out,
        (int16_t*) &shift_scale->data.i16[0],
        (int16_t*) &shift_scale->data.i16[scales_offset]
    );

  return kTfLiteOk;
}

TfLiteRegistration* Register_Conv2DShallowinDeepoutFinal() {
    static TfLiteRegistration r = {nullptr, nullptr, Conv2DShallowinDeepoutPrepare, Conv2DShallowinDeepoutEval};
    return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_CONV2D_SHALLOWOUT_DEEPOUT_H_
