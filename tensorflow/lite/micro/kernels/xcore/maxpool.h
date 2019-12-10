#ifndef XCORE_OPS_MAXPOOL_H_
#define XCORE_OPS_MAXPOOL_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

extern "C" {
    #include "nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {

TfLiteStatus MaxPool2DDeepPrepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    return kTfLiteOk;
}

TfLiteStatus MaxPool2DDeepEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = GetInput(context, node, 0);

    int32_t height = input->dims->data[1];
    int32_t width = input->dims->data[2];
    int32_t C_in = input->dims->data[3];

    TfLiteTensor* output = GetOutput(context, node, 0);

    maxpool2d_deep(
        input->data.int8,
        output->data.int8,
        height,
        width,
        C_in
    );

  return kTfLiteOk;
}

TfLiteRegistration* Register_MaxPool2DDeep() {
    static TfLiteRegistration r = {nullptr, nullptr, MaxPool2DDeepPrepare, MaxPool2DDeepEval};
    return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_MAXPOOL_H_
