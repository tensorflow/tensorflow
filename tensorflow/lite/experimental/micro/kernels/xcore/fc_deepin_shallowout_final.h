#ifndef XCORE_OPS_FC_DEEPIN_SHALLOWOUT_FINAL_H_
#define XCORE_OPS_FC_DEEPIN_SHALLOWOUT_FINAL_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

extern "C" {
    #include "nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {

TfLiteStatus FCDeepinShallowoutFinalPrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    return kTfLiteOk;
}

TfLiteStatus FCDeepinShallowoutFinalEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = GetInput(context, node, 0);
    const TfLiteTensor* weights = GetInput(context, node, 1);
    const TfLiteTensor* biases = GetInput(context, node, 2);
    const TfLiteTensor* shift_scale = GetInput(context, node, 3);

    int32_t C_out = weights->dims->data[0];
    int32_t C_in = weights->dims->data[1];
    int32_t scales_offset = C_out;

    TfLiteTensor* output = GetOutput(context, node, 0);

    fc_deepin_shallowout_lin(
        weights->data.int8,
        biases->data.i32,
        input->data.int8,
        output->data.i16,
        C_out,
        C_in,
        (uint16_t*) &shift_scale->data.i16[0],
        (int16_t*) &shift_scale->data.i16[scales_offset]
    );

  return kTfLiteOk;
}

TfLiteRegistration* Register_FCDeepinShallowoutFinal() {
    static TfLiteRegistration r = {nullptr, nullptr, FCDeepinShallowoutFinalPrepare, FCDeepinShallowoutFinalEval};
    return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_FC_DEEPIN_SHALLOWOUT_FINAL_H_
