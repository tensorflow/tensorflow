#include <iostream>

#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

    static constexpr int unpadded_shape_array_len = 4;

    typedef struct {
        padding_mode_t padding_mode;
        int32_t unpadded_shape[unpadded_shape_array_len]; // values represent [C_out, K_h, K_w, C_in]
        nn_conv2d_sido_params_t kernel_params;
    } SIDO_UserData;


    typedef struct {
        padding_mode_t padding_mode;
        nn_conv2d_dido_params_t kernel_params;
    } DIDO_UserData;


    static void parse_sido_options(const char* buffer, size_t length, SIDO_UserData *data)
    {
        const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
        // std::cout << flexbuffers::GetRoot(buffer_t, length).ToString() << std::endl;
        auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

        auto keys = map.Keys();
        auto values = map.Values();
        for (int i=0; i<map.size(); ++i)
        {
            std::string key(keys[i].ToString());
            
            if (key.compare("padding") == 0)
            {
                std::string padding_mode_str = values[i].ToString();
                if (padding_mode_str.compare("VALID") == 0) 
                    data->padding_mode = PADDING_VALID;
                else 
                    data->padding_mode = PADDING_SAME;
            }
            else if (key.compare("unpadded_shape") == 0)
            {
                auto vec = values[i].AsVector();
                for (int j=0; j<unpadded_shape_array_len; ++j)
                    data->unpadded_shape[j] = vec[j].AsInt32();
            }
        }
    }


    static void parse_dido_options(const char* buffer, size_t length, DIDO_UserData *data)
    {
        const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
        auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

        auto keys = map.Keys();
        auto values = map.Values();
        for (int i=0; i<map.size(); ++i)
        {
            std::string key(keys[i].ToString());
            
            if (key.compare("padding") == 0)
            {
                std::string padding_mode_str = values[i].ToString();
                if (padding_mode_str.compare("VALID") == 0) 
                    data->padding_mode = PADDING_VALID;
                else 
                    data->padding_mode = PADDING_SAME;
            }
        }
    }

    //**************************************
    //**************************************
    //**************************************
    // Shallowin_Deepout
    //**************************************
    //**************************************
    //**************************************
    void* Init_SIDO(TfLiteContext* context, const char* buffer, size_t length) 
    {
        auto* user_data = new SIDO_UserData();

        if (buffer)
            parse_sido_options(buffer, length, user_data);

        return user_data;
    }

    void Free_SIDO(TfLiteContext* context, void* buffer) {
        auto* user_data = reinterpret_cast<SIDO_UserData*>(buffer);
        nn_conv2d_sido_params_t* kernel_params = (nn_conv2d_sido_params_t*) &user_data->kernel_params;

        conv2d_shallowin_deepout_deinit(kernel_params);
        delete user_data;
    }


    TfLiteStatus Prepare_SIDO(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* biases = GetInput(context, node, 2);
        const TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<SIDO_UserData*>(node->user_data);
        nn_conv2d_sido_params_t* kernel_params = (nn_conv2d_sido_params_t*) &user_data->kernel_params;

        if (kernel_params)
        {
            // need to deinit first because something significant has changes with the operator
            conv2d_shallowin_deepout_deinit(kernel_params);
        }

        nn_conv2d_init_params_t init_params;
        nn_conv2d_region_params_t region_params;

        init_params.X_height = input->dims->data[1];
        init_params.X_width = input->dims->data[2];
        init_params.K_h = user_data->unpadded_shape[1];
        init_params.K_w = user_data->unpadded_shape[2];
        init_params.C_in = input->dims->data[3];
        init_params.C_out = user_data->unpadded_shape[0];
        init_params.pad_mode = user_data->padding_mode;
        init_params.zero_point = input->params.zero_point;

        region_params.top = 0;
        region_params.left = 0;
        region_params.rows = output->dims->data[1];
        region_params.cols = output->dims->data[2];

        // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
        //         TfLiteTensor to replace TfLiteQuantizationParams params
        conv2d_shallowin_deepout_init(
            kernel_params,
            &init_params,
            &region_params,
            weights->data.int8,
            (data16_t*) biases->data.i16
        );

        return kTfLiteOk;
    }


    TfLiteStatus Eval_SIDO(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* shift_scale = GetInput(context, node, 3);
        TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<SIDO_UserData*>(node->user_data);
        const nn_conv2d_sido_params_t *kernel_params = &user_data->kernel_params;
        unsigned block_count = kernel_params->block_count;
        int32_t scales_offset = user_data->unpadded_shape[0]; // C_out

        for (unsigned i=0; i < block_count; i++)
        {
            conv2d_shallowin_deepout_block(
                output->data.int8, // Y
                kernel_params,
                &kernel_params->blocks[i],
                input->data.int8, // X,
                weights->data.int8, // K
                (int16_t*) &shift_scale->data.i16[0], // shifts
                (int16_t*) &shift_scale->data.i16[scales_offset] // scales
            );
        }

        return kTfLiteOk;
    }

    //**************************************
    //**************************************
    //**************************************
    // Conv2D_Deepin_Deepout
    //**************************************
    //**************************************
    //**************************************

    void* Init_DIDO(TfLiteContext* context, const char* buffer, size_t length) 
    {
        auto* user_data = new DIDO_UserData();
        if (buffer)
            parse_dido_options(buffer, length, user_data);

        return user_data;
    }

    void Free_DIDO(TfLiteContext* context, void* buffer) {
        auto* user_data = reinterpret_cast<DIDO_UserData*>(buffer);
        nn_conv2d_dido_params_t* kernel_params = (nn_conv2d_dido_params_t*) &user_data->kernel_params;

        conv2d_deepin_deepout_deinit(kernel_params);
        delete user_data;
    }

    TfLiteStatus Prepare_DIDO(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* biases = GetInput(context, node, 2);
        const TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<DIDO_UserData*>(node->user_data);
        nn_conv2d_dido_params_t* kernel_params = (nn_conv2d_dido_params_t*) &user_data->kernel_params;

        if (kernel_params)
        {
            // need to deinit first because something significant has changes with the operator
            conv2d_deepin_deepout_deinit(kernel_params);
        }

        nn_conv2d_init_params_t init_params;
        nn_conv2d_region_params_t  region_params;

        init_params.X_height = input->dims->data[1];
        init_params.X_width = input->dims->data[2];
        init_params.K_h = weights->dims->data[1];
        init_params.K_w = weights->dims->data[2];
        init_params.C_in = weights->dims->data[3] * weights->dims->data[5];
        init_params.C_out = weights->dims->data[0] * weights->dims->data[4];
        init_params.pad_mode = user_data->padding_mode;
        init_params.zero_point = input->params.zero_point;

        region_params.top = 0;
        region_params.left = 0;
        region_params.rows = output->dims->data[1];
        region_params.cols = output->dims->data[2];

        // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
        //         TfLiteTensor to replace TfLiteQuantizationParams params
        conv2d_deepin_deepout_init(
            kernel_params,
            &init_params,
            &region_params,
            weights->data.int8,
            (data16_t*) biases->data.i16
        );

        return kTfLiteOk;
    }

    TfLiteStatus Eval_DIDO(TfLiteContext* context, TfLiteNode* node) {
        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* shift_scale = GetInput(context, node, 3);
        TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<DIDO_UserData*>(node->user_data);
        const nn_conv2d_dido_params_t *kernel_params = &user_data->kernel_params;
        unsigned block_count = kernel_params->block_count;
        int32_t scales_offset = weights->dims->data[0] * weights->dims->data[4]; // C_out

        for (unsigned i=0; i < block_count; i++)
        {
            conv2d_deepin_deepout_block(
                output->data.int8, // Y
                kernel_params,
                &kernel_params->blocks[i],
                input->data.int8, // X,
                weights->data.int8, // K
                (int16_t*) &shift_scale->data.i16[0], // shifts
                (int16_t*) &shift_scale->data.i16[scales_offset] // scales
            );
        }

        return kTfLiteOk;
    }

}  // conv


TfLiteRegistration* Register_Conv_DIDO() {
    static TfLiteRegistration r = {
        conv::Init_DIDO,
        conv::Free_DIDO,
        conv::Prepare_DIDO,
        conv::Eval_DIDO
    };
    return &r;
}

TfLiteRegistration* Register_Conv_SIDO() {
    static TfLiteRegistration r = {
        conv::Init_SIDO,
        conv::Free_SIDO,
        conv::Prepare_SIDO,
        conv::Eval_SIDO
    };
    return &r;
}


}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

