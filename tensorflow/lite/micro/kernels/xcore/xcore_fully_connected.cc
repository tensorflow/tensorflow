#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace fully_connected {

    typedef struct {
        nn_fully_connected_plan_t plan;
    } UserData;

    void* Init_16(TfLiteContext* context, const char* buffer, size_t length) 
    {
        auto* user_data = new UserData();

        return user_data;
    }

    void Free_16(TfLiteContext* context, void* buffer) {
        auto* user_data = reinterpret_cast<UserData*>(buffer);

        delete user_data;
    }


    TfLiteStatus Prepare_16(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        const TfLiteTensor* weights = GetInput(context, node, 1);
        int32_t C_in = weights->dims->data[1];
        int32_t C_out = weights->dims->data[0];

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        fully_connected_init(&user_data->plan, C_in, C_out);

        return kTfLiteOk;
    }


    TfLiteStatus Eval_16(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* bias_shift_scale = GetInput(context, node, 2);

        TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        fully_connected_16(
            output->data.i16,
            weights->data.int8,
            input->data.int8,
            (data16_t*) bias_shift_scale->data.i16,
            &user_data->plan
        );

        return kTfLiteOk;
    }


}  // namespace fully_connected


TfLiteRegistration* Register_FullyConnected_16() {
    static TfLiteRegistration r = {
        fully_connected::Init_16,
        fully_connected::Free_16,
        fully_connected::Prepare_16,
        fully_connected::Eval_16
    };
    return &r;
}


}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
