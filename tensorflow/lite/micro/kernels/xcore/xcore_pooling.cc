#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace max_pool {

    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        return kTfLiteOk;
    }

    TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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

} //namespace max_pool

namespace avg_pool {

    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        return kTfLiteOk;
    }

    TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);

        int32_t height = input->dims->data[1];
        int32_t width = input->dims->data[2];
        int32_t C_in = input->dims->data[3];

        TfLiteTensor* output = GetOutput(context, node, 0);

        avgpool2d_deep(
            input->data.int8,
            output->data.int8,
            height,
            width,
            C_in
        );

        return kTfLiteOk;
    }

}  // namespace avg_pool


TfLiteRegistration* Register_MaxPool() {
    static TfLiteRegistration r = {
        nullptr,
        nullptr,
        max_pool::Prepare,
        max_pool::Eval
    };
    return &r;
}


TfLiteRegistration* Register_AvgPool() {
    static TfLiteRegistration r = {
        nullptr,
        nullptr,
        avg_pool::Prepare,
        avg_pool::Eval
    };
    return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
