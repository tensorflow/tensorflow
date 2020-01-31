#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace type_conversions {

    TfLiteStatus Prepare_Requantize_16_to_8(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        return kTfLiteOk;
    }

    TfLiteStatus Eval_Requantize_16_to_8(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        TfLiteTensor* output = GetOutput(context, node, 0);
        int32_t n = input->bytes / sizeof(int16_t);

        requantize_16_to_8(
            output->data.int8,
            input->data.i16,
            n
        );

        return kTfLiteOk;
    }

}  // namespace type_conversions


TfLiteRegistration* Register_Requantize_16_to_8() {
    static TfLiteRegistration r = {
        nullptr,
        nullptr,
        type_conversions::Prepare_Requantize_16_to_8,
        type_conversions::Eval_Requantize_16_to_8
    };
    return &r;
}


}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
