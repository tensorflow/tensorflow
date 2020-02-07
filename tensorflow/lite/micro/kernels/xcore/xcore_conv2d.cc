#include <iostream>

#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_par.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv2d {

    typedef union ParamsPtrUnion {
        nn_conv2d_sido_params_t *sido;
        nn_conv2d_dido_params_t *dido;
    } ParamsUnion;


    typedef struct {
        padding_mode_t padding_mode;
        int32_t C_in;
        int32_t C_out;
        int32_t K_h;
        int32_t K_w;
        ParPlan par_plan;
        ParamsPtrUnion params;
    } UserData;


    static void parse_options(const char* buffer, size_t length, UserData *data)
    {
        const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
        // std::cout << flexbuffers::GetRoot(buffer_t, length).ToString() << std::endl;
        auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

        auto keys = map.Keys();
        auto values = map.Values();
        for (int i = 0; i < map.size(); ++i)
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
                auto vec = values[i].AsVector(); // values represent [C_out, K_h, K_w, C_in]
                data->C_out = vec[0].AsInt32();
                data->K_h = vec[1].AsInt32();
                data->K_w = vec[2].AsInt32();
                data->C_in = vec[3].AsInt32();
            }
            else if (key.compare("par_plan") == 0)
            {
                auto jobs = values[i].AsVector();
                data->par_plan.n_regions = jobs.size();
                for (int i=0; i<jobs.size(); ++i)
                {
                    auto region = jobs[i].AsVector();
                    data->par_plan.regions[i].top = region[0].AsInt32();
                    data->par_plan.regions[i].left = region[1].AsInt32();
                    data->par_plan.regions[i].rows = region[2].AsInt32();
                    data->par_plan.regions[i].cols = region[3].AsInt32();
                }

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
        auto* user_data = new UserData();
        user_data->params.sido = new nn_conv2d_sido_params_t();

        if (buffer)
            parse_options(buffer, length, user_data);

        return user_data;
    }

    void Free_SIDO(TfLiteContext* context, void* buffer) {
        auto* user_data = reinterpret_cast<UserData*>(buffer);

        conv2d_shallowin_deepout_deinit(user_data->params.sido);
        delete user_data;
    }


    TfLiteStatus Prepare_SIDO(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* biases = GetInput(context, node, 2);
        const TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        if (user_data->params.sido->blocks)
        {
            // need to deinit first because something significant has changes with the operator
            conv2d_shallowin_deepout_deinit(user_data->params.sido);
        }

        nn_conv2d_init_params_t init_params;
        nn_conv2d_region_params_t region_params;

        init_params.X_height = input->dims->data[1];
        init_params.X_width = input->dims->data[2];
        init_params.K_h = user_data->K_h;
        init_params.K_w = user_data->K_w;
        init_params.C_in = input->dims->data[3]; // number of channels after padding
        init_params.C_out = user_data->C_out;
        init_params.pad_mode = user_data->padding_mode;
        init_params.zero_point = input->params.zero_point;

        region_params.top = 0;
        region_params.left = 0;
        region_params.rows = output->dims->data[1];
        region_params.cols = output->dims->data[2];

        // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
        //         TfLiteTensor to replace TfLiteQuantizationParams params
        conv2d_shallowin_deepout_init(
            user_data->params.sido,
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

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        conv2d_shallowin_deepout(
            output->data.int8, // Y
            user_data->params.sido,
            input->data.int8, // X,
            weights->data.int8, // K
            (int16_t*) shift_scale->data.i16 // shifts & scales
        );

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
        auto* user_data = new UserData();
        user_data->params.dido = new nn_conv2d_dido_params_t();

        if (buffer)
            parse_options(buffer, length, user_data);

        return user_data;
    }

    void Free_DIDO(TfLiteContext* context, void* buffer) {
        auto* user_data = reinterpret_cast<UserData*>(buffer);

        conv2d_deepin_deepout_deinit(user_data->params.dido);
        delete user_data;
    }

    TfLiteStatus Prepare_DIDO(TfLiteContext* context, TfLiteNode* node) {
        TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
        TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

        const TfLiteTensor* input = GetInput(context, node, 0);
        const TfLiteTensor* weights = GetInput(context, node, 1);
        const TfLiteTensor* biases = GetInput(context, node, 2);
        const TfLiteTensor* output = GetOutput(context, node, 0);

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        // set param values not parsed from custom options
        user_data->K_h = weights->dims->data[1];
        user_data->K_w = weights->dims->data[2];
        user_data->C_in = weights->dims->data[3] * weights->dims->data[5];
        user_data->C_out = weights->dims->data[0] * weights->dims->data[4];

        if (user_data->params.dido->blocks)
        {
            // need to deinit first because something significant has changes with the operator
            conv2d_deepin_deepout_deinit(user_data->params.dido);
        }

        nn_conv2d_init_params_t init_params;
        nn_conv2d_region_params_t  region_params;

        init_params.X_height = input->dims->data[1];
        init_params.X_width = input->dims->data[2];
        init_params.K_h = user_data->K_h;
        init_params.K_w = user_data->K_w;
        init_params.C_in = user_data->C_in;
        init_params.C_out = user_data->C_out;
        init_params.pad_mode = user_data->padding_mode;
        init_params.zero_point = input->params.zero_point;

        region_params.top = 0;
        region_params.left = 0;
        region_params.rows = output->dims->data[1];
        region_params.cols = output->dims->data[2];

        // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
        //         TfLiteTensor to replace TfLiteQuantizationParams params
        conv2d_deepin_deepout_init(
            user_data->params.dido,
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

        auto* user_data = reinterpret_cast<UserData*>(node->user_data);

        conv2d_deepin_deepout(
            output->data.int8, // Y
            user_data->params.dido,
            input->data.int8, // X,
            weights->data.int8, // K
            (int16_t*) shift_scale->data.i16 // shifts & scales
        );

        return kTfLiteOk;
    }

}  // conv2d


TfLiteRegistration* Register_Conv2D_DIDO() {
    static TfLiteRegistration r = {
        conv2d::Init_DIDO,
        conv2d::Free_DIDO,
        conv2d::Prepare_DIDO,
        conv2d::Eval_DIDO
    };
    return &r;
}

TfLiteRegistration* Register_Conv2D_SIDO() {
    static TfLiteRegistration r = {
        conv2d::Init_SIDO,
        conv2d::Free_SIDO,
        conv2d::Prepare_SIDO,
        conv2d::Eval_SIDO
    };
    return &r;
}


}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

