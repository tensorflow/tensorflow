#include <iostream>
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pooling {

    typedef struct {
        int32_t pool_h;
        int32_t pool_w;
        int32_t stride_h;
        int32_t stride_w;
    } PoolingOptions;

    template<int N, class T>
    T unpack(const uint8_t *buffer) {
        T retval =0;
        for (int i=0; i<N; ++i)
            retval |= buffer[i] << (8 * i);
        return retval;
    }

    static void parse_options(const char* buffer, size_t length, PoolingOptions *options)
    {
        const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
        auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

        auto keys = map.Keys();
        auto values = map.Values();
        for (int i = 0; i < map.size(); ++i)
        {
            std::string key(keys[i].ToString());

            if (key.compare("pool") == 0)
            {
                auto vec = values[i].AsVector(); // values represent [pool_h, pool_w]
                options->pool_h = vec[0].AsInt32();
                options->pool_w = vec[1].AsInt32();
            }
            else if (key.compare("stride") == 0)
            {
                auto vec = values[i].AsVector(); // values represent [stride_h, stride_w]
                options->stride_h = vec[0].AsInt32();
                options->stride_w = vec[1].AsInt32();
            }
        }
        std::cout << std::endl;
    }

    //**************************************
    //**************************************
    //**************************************
    // MaxPool
    //**************************************
    //**************************************
    //**************************************
    namespace maxpool {

        typedef struct {
            PoolingOptions options;
            nn_window_op_plan_t plan;
        } OpData;

        void* Init2D(TfLiteContext* context, const char* buffer, size_t length) {
            OpData* op_data = nullptr;
            context->AllocatePersistentBuffer(context, sizeof(OpData), (void**) &op_data);

            if (buffer)
                parse_options(buffer, length, &op_data->options);

            return op_data;
        }

        TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
            TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
            TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
            const TfLiteTensor* input = GetInput(context, node, 0);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* op_data = reinterpret_cast<OpData*>(node->user_data);

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
                                    op_data->options.pool_h, op_data->options.pool_w,
                                    op_data->options.stride_h, op_data->options.stride_w);

            maxpool2d_init(&op_data->plan, &params_in, &params_out, &config);

            return kTfLiteOk;
        }

        TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input = GetInput(context, node, 0);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* op_data = reinterpret_cast<OpData*>(node->user_data);

            maxpool2d(
                output->data.int8,
                input->data.int8,
                &op_data->plan
            );

            return kTfLiteOk;
        }

    } //namespace maxpool

    //**************************************
    //**************************************
    //**************************************
    // AvgPool
    //**************************************
    //**************************************
    //**************************************
    namespace avgpool {

        typedef struct {
            PoolingOptions options;
            nn_avgpool2d_plan_t plan;
        } OpData;

        void* Init2D(TfLiteContext* context, const char* buffer, size_t length) {
            OpData* op_data = nullptr;
            context->AllocatePersistentBuffer(context, sizeof(OpData), (void**) &op_data);

            if (buffer)
                parse_options(buffer, length, &op_data->options);

            return op_data;
        }

        TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
            TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
            TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
            const TfLiteTensor* input = GetInput(context, node, 0);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* op_data = reinterpret_cast<OpData*>(node->user_data);

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
                                    op_data->options.pool_h, op_data->options.pool_w,
                                    op_data->options.stride_h, op_data->options.stride_w);

            avgpool2d_init(&op_data->plan, &params_in, &params_out, &config);

            return kTfLiteOk;
        }

        TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input = GetInput(context, node, 0);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* op_data = reinterpret_cast<OpData*>(node->user_data);

            avgpool2d(
                output->data.int8,
                input->data.int8,
                &op_data->plan
            );

            return kTfLiteOk;
        }

    }  // namespace avgpool


    //**************************************
    //**************************************
    //**************************************
    // AvgPool_Global
    //**************************************
    //**************************************
    //**************************************
    namespace avgpool_global {

        typedef struct {
            int32_t bias;
            uint32_t scale;
            uint32_t shift;
        } OpData;

        void* Init2D(TfLiteContext* context, const char* buffer, size_t length) {
            OpData* op_data = nullptr;
            context->AllocatePersistentBuffer(context, sizeof(OpData), (void**) &op_data);

            return op_data;
        }

        TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
            TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
            TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
            
            const TfLiteTensor* bss = GetInput(context, node, 1);

            auto* op_data = reinterpret_cast<OpData*>(node->user_data);

            op_data->bias= unpack<4, int32_t>(&bss->data.uint8[0]);
            op_data->scale = unpack<1, uint32_t>(&bss->data.uint8[4]);
            op_data->shift = unpack<2, uint32_t>(&bss->data.uint8[5]);

            return kTfLiteOk;
        }

        TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input = GetInput(context, node, 0);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* op_data = reinterpret_cast<OpData*>(node->user_data);

            uint32_t X_height = input->dims->data[1];
            uint32_t X_width = input->dims->data[2];
            uint32_t C_in = input->dims->data[3];

            avgpool2d_global(
                output->data.int8,
                input->data.int8,
                X_height,
                X_width,
                C_in,
                op_data->bias,
                op_data->shift,
                op_data->scale
            );

            return kTfLiteOk;
        }

    }  // namespace avgpool_global

}  // namespace pooling

TfLiteRegistration* Register_MaxPool2D() {
    static TfLiteRegistration r = {
        pooling::maxpool::Init2D,
        nullptr,
        pooling::maxpool::Prepare2D,
        pooling::maxpool::Eval2D
    };
    return &r;
}


TfLiteRegistration* Register_AvgPool2D() {
    static TfLiteRegistration r = {
        pooling::avgpool::Init2D,
        nullptr,
        pooling::avgpool::Prepare2D,
        pooling::avgpool::Eval2D
    };
    return &r;
}

TfLiteRegistration* Register_AvgPool2D_Global() {
    static TfLiteRegistration r = {
        pooling::avgpool_global::Init2D,
        nullptr,
        pooling::avgpool_global::Prepare2D,
        pooling::avgpool_global::Eval2D
    };
    return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
