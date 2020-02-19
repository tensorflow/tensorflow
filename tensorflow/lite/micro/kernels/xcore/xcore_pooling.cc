#include <iostream>
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

template<int N, class T>
T unpack(const uint8_t *buffer) {
    T retval = 0;
    for (int i=0; i<N; ++i)
        retval |= buffer[i] << (8*i);
    return retval;
}

namespace maxpool {

    TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        return kTfLiteOk;
    }

    TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
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

} //namespace maxpool

namespace avgpool {

    typedef struct {
        uint32_t pool_h;
        uint32_t pool_w;
        uint32_t stride_h;
        uint32_t stride_w;
        nn_avgpool2d_plan_t plan;
    } UserData;

    static void parse_options(const char* buffer, size_t length, UserData *data)
    {
        const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
        auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

        auto keys = map.Keys();
        auto values = map.Values();
        for (int i = 0; i < map.size(); ++i)
        {
            std::string key(keys[i].ToString());
            
            if (key.compare("pool_h") == 0)
                data->pool_h = values[0].AsInt32();
            else if (key.compare("pool_w") == 0)
                data->pool_w = values[1].AsInt32();
            else if (key.compare("stride_h") == 0)
                data->stride_h = values[2].AsInt32();
            else if (key.compare("stride_w") == 0)
                data->stride_w = values[3].AsInt32();
        }
        std::cout << std::endl;
    }


    void* Init2D(TfLiteContext* context, const char* buffer, size_t length) 
    {
        auto* user_data = new UserData();

        if (buffer)
            parse_options(buffer, length, user_data);

        return user_data;
    }

    void Free2D(TfLiteContext* context, void* buffer) {
        auto* user_data = reinterpret_cast<UserData*>(buffer);

        delete user_data;
    }

    TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
        const TfLiteTensor* input = GetInput(context, node, 0);
        TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        nn_image_params_t params_in;
        params_in.height = input->dims->data[1];
        params_in.width = input->dims->data[2];
        params_in.channels = input->dims->data[3];

        nn_image_params_t params_out;
        params_out.height = output->dims->data[1];
        params_out.width = output->dims->data[2];
        params_out.channels = output->dims->data[3];

        nn_window_op_config_t config;
        nn_window_op_config_simple(&config, &params_in, &params_out,
                                   user_data->pool_h, user_data->pool_w,
                                   user_data->stride_h, user_data->stride_w);

        avgpool2d_init(&user_data->plan, &params_in, &params_out, &config);

        return kTfLiteOk;
    }

    TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        avgpool2d(
            output->data.int8,
            input->data.int8,
            &user_data->plan
        );

        return kTfLiteOk;
    }

}  // namespace avgpool



namespace avgpool_global {

    typedef struct {
        int32_t bias;
        uint32_t scale;
        uint32_t shift;
    } UserData;

    void* Init2D(TfLiteContext* context, const char* buffer, size_t length) 
    {
        auto* user_data = new UserData();

        return user_data;
    }

    void Free2D(TfLiteContext* context, void* buffer) {
        auto* user_data = reinterpret_cast<UserData*>(buffer);

        delete user_data;
    }

    TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
        
        const TfLiteTensor* bss = GetInput(context, node, 1);

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        user_data->bias = unpack<4, int32_t>(&bss->data.uint8[0]);
        user_data->scale = unpack<1, uint32_t>(&bss->data.uint8[4]);
        user_data->shift = unpack<2, uint32_t>(&bss->data.uint8[5]);

        return kTfLiteOk;
    }

    TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        uint32_t X_height = input->dims->data[1];
        uint32_t X_width = input->dims->data[2];
        uint32_t C_in = input->dims->data[3];

        avgpool2d_global(
            output->data.int8,
            input->data.int8,
            X_height,
            X_width,
            C_in,
            user_data->bias,
            user_data->shift,
            user_data->scale
        );

        return kTfLiteOk;
    }

}  // namespace avgpool_global


TfLiteRegistration* Register_MaxPool2D() {
    static TfLiteRegistration r = {
        nullptr,
        nullptr,
        maxpool::Prepare2D,
        maxpool::Eval2D
    };
    return &r;
}


TfLiteRegistration* Register_AvgPool2D() {
    static TfLiteRegistration r = {
        avgpool::Init2D,
        avgpool::Free2D,
        avgpool::Prepare2D,
        avgpool::Eval2D
    };
    return &r;
}

TfLiteRegistration* Register_AvgPool2D_Global() {
    static TfLiteRegistration r = {
        avgpool_global::Init2D,
        avgpool_global::Free2D,
        avgpool_global::Prepare2D,
        avgpool_global::Eval2D
    };
    return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
