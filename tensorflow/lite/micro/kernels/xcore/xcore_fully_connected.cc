#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace fully_connected {

    // TfLiteStatus Prepare_16(TfLiteContext* context, TfLiteNode* node) {
    //     TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
    //     TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    //     return kTfLiteOk;
    // }


    // TfLiteStatus Eval_16(TfLiteContext* context, TfLiteNode* node) {
    //     const TfLiteTensor* input = GetInput(context, node, 0);
    //     const TfLiteTensor* weights = GetInput(context, node, 1);
    //     const TfLiteTensor* bias_shift_scale = GetInput(context, node, 2);

    //     TfLiteTensor* output = GetOutput(context, node, 0);

    //     int32_t C_in = weights->dims->data[1];
    //     int32_t C_out = weights->dims->data[0];

    //     fully_connected_16(
    //         output->data.i16,
    //         weights->data.int8,
    //         input->data.int8,
    //         (data16_t*) bias_shift_scale->data.i16,
    //         C_in,
    //         C_out
    //     );

    //     return kTfLiteOk;
    // }


    TfLiteStatus Prepare_DIAO(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        return kTfLiteOk;
    }


    TfLiteStatus Eval_DIAO(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* biases = GetInput(context, node, 2);
        const TfLiteTensor* shift_scale = GetInput(context, node, 3);

        int32_t C_out = weights->dims->data[0];
        int32_t C_in = weights->dims->data[1];
        int32_t scales_offset = C_out;

        TfLiteTensor* output = GetOutput(context, node, 0);

        fc_deepin_shallowout_16(
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

}  // namespace fully_connected


// TfLiteRegistration* Register_FullyConnected_16() {
//     static TfLiteRegistration r = {
//         nullptr,
//         nullptr,
//         fully_connected::Prepare_16,
//         fully_connected::Eval_16
//     };
//     return &r;
// }


TfLiteRegistration* Register_FullyConnected_DIAO() {
    static TfLiteRegistration r = {
        nullptr,
        nullptr,
        fully_connected::Prepare_DIAO,
        fully_connected::Eval_DIAO
    };
    return &r;
}


}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
