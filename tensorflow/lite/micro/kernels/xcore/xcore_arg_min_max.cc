#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace arg_max {

    TfLiteStatus Prepare_16(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        return kTfLiteOk;
    }

    TfLiteStatus Eval_16(TfLiteContext* context, TfLiteNode* node) {
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

}  // namespace arg_max


TfLiteRegistration* Register_ArgMax_16() {
    static TfLiteRegistration r = {
        nullptr,
        nullptr,
        arg_max::Prepare_16,
        arg_max::Eval_16
    };
    return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
