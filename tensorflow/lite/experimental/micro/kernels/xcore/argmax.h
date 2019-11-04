#ifndef XCORE_OPS_ARGMAX_H_
#define XCORE_OPS_ARGMAX_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

extern "C" {
    #include "nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {

TfLiteStatus ArgMax16Prepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    return kTfLiteOk;
}

TfLiteStatus ArgMax16Eval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = GetInput(context, node, 0);

    int32_t N = input->bytes / sizeof(int16_t);

    TfLiteTensor* output = GetOutput(context, node, 0);

    argmax_16(
        input->data.i16,
        output->data.i32,
        N
    );

    return kTfLiteOk;
}

TfLiteRegistration* Register_ArgMax16() {
    static TfLiteRegistration r = {nullptr, nullptr, ArgMax16Prepare, ArgMax16Eval};
    return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_ARGMAX_H_
