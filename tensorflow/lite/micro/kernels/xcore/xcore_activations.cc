#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace activations {

    TfLiteStatus Prepare_Lookup_8(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        return kTfLiteOk;
    }

    TfLiteStatus Eval_Lookup_8(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* lut = GetInput(context, node, 1);
        TfLiteTensor* output = GetOutput(context, node, 0);
        int32_t length = input->bytes / sizeof(uint8_t);

        lookup8(
            output->data.uint8,
            input->data.uint8,
            lut->data.uint8,
            length
        );

        return kTfLiteOk;
    }

}  // namespace activations


TfLiteRegistration* Register_Lookup_8() {
    static TfLiteRegistration r = {
        nullptr,
        nullptr,
        activations::Prepare_Lookup_8,
        activations::Eval_Lookup_8
    };
    return &r;
}


}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
