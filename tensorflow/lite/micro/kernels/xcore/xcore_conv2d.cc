#include <iostream>

#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_par.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

    typedef struct {
        padding_mode_t padding_mode;
        int32_t C_in;
        int32_t C_out;
        int32_t K_h;
        int32_t K_w;
        ParPlan par_plan;
    } Conv2DOptions;

    static void parse_options(const char* buffer, size_t length, Conv2DOptions *options)
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
                    options->padding_mode = PADDING_VALID;
                else 
                    options->padding_mode = PADDING_SAME;
            }
            else if (key.compare("unpadded_shape") == 0)
            {
                auto vec = values[i].AsVector(); // values represent [C_out, K_h, K_w, C_in]
                options->C_out = vec[0].AsInt32();
                options->K_h = vec[1].AsInt32();
                options->K_w = vec[2].AsInt32();
                options->C_in = vec[3].AsInt32();
            }
            else if (key.compare("par_plan") == 0)
            {
                auto jobs = values[i].AsVector();
                options->par_plan.n_regions = jobs.size();
                for (int i=0; i<jobs.size(); ++i)
                {
                    auto region = jobs[i].AsVector();
                    options->par_plan.regions[i].top = region[0].AsInt32();
                    options->par_plan.regions[i].left = region[1].AsInt32();
                    options->par_plan.regions[i].rows = region[2].AsInt32();
                    options->par_plan.regions[i].cols = region[3].AsInt32();
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
    namespace sido {

        typedef struct {
            Conv2DOptions options;
            nn_conv2d_sido_params_t *params;
        } UserData;

        void* Init2D(TfLiteContext* context, const char* buffer, size_t length) 
        {
            auto* user_data = new UserData();
            user_data->params = new nn_conv2d_sido_params_t();

            if (buffer)
                parse_options(buffer, length, &user_data->options);

            return user_data;
        }

        void Free2D(TfLiteContext* context, void* buffer) {
            auto* user_data = reinterpret_cast<UserData*>(buffer);

            conv2d_shallowin_deepout_deinit(user_data->params);
            delete user_data;
        }


        TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
            TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
            TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

            const TfLiteTensor* input = GetInput(context, node, 0);
            const TfLiteTensor* weights = GetInput(context, node, 1);
            const TfLiteTensor* biases = GetInput(context, node, 2);
            const TfLiteTensor* output = GetOutput(context, node, 0);

            auto* user_data = reinterpret_cast<UserData*>(node->user_data);

            if (user_data->params->blocks)
            {
                // need to deinit first because something significant has changes with the operator
                conv2d_shallowin_deepout_deinit(user_data->params);
            }

            nn_conv2d_init_params_t init_params;
            nn_conv2d_region_params_t region_params;

            init_params.X_height = input->dims->data[1];
            init_params.X_width = input->dims->data[2];
            init_params.K_h = user_data->options.K_h;
            init_params.K_w = user_data->options.K_w;
            init_params.C_in = input->dims->data[3]; // number of channels after padding
            init_params.C_out = user_data->options.C_out;
            init_params.pad_mode = user_data->options.padding_mode;
            init_params.zero_point = input->params.zero_point;

            region_params.top = 0;
            region_params.left = 0;
            region_params.rows = output->dims->data[1];
            region_params.cols = output->dims->data[2];

            // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
            //         TfLiteTensor to replace TfLiteQuantizationParams params
            conv2d_shallowin_deepout_init(
                user_data->params,
                &init_params,
                &region_params,
                weights->data.int8,
                (data16_t*) biases->data.i16
            );

            return kTfLiteOk;
        }


        TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input = GetInput(context, node, 0);
            const TfLiteTensor* weights = GetInput(context, node, 1);
            const TfLiteTensor* shift_scale = GetInput(context, node, 3);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* user_data = reinterpret_cast<UserData*>(node->user_data);

            conv2d_shallowin_deepout(
                output->data.int8, // Y
                user_data->params,
                input->data.int8, // X,
                weights->data.int8, // K
                (int16_t*) shift_scale->data.i16 // shifts & scales
            );

            return kTfLiteOk;
        }

    } //namespace sido


    //**************************************
    //**************************************
    //**************************************
    // Conv2D_Deepin_Deepout
    //**************************************
    //**************************************
    //**************************************
    namespace dido {

        typedef struct {
            Conv2DOptions options;
            nn_conv2d_dido_params_t *params;
        } UserData;

        void* Init2D(TfLiteContext* context, const char* buffer, size_t length) 
        {
            auto* user_data = new UserData();
            user_data->params = new nn_conv2d_dido_params_t();

            if (buffer)
                parse_options(buffer, length, &user_data->options);

            return user_data;
        }

        void Free2D(TfLiteContext* context, void* buffer) {
            auto* user_data = reinterpret_cast<UserData*>(buffer);

            conv2d_deepin_deepout_deinit(user_data->params);
            delete user_data;
        }

        TfLiteStatus Prepare2D(TfLiteContext* context, TfLiteNode* node) {
            TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
            TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

            const TfLiteTensor* input = GetInput(context, node, 0);
            const TfLiteTensor* weights = GetInput(context, node, 1);
            const TfLiteTensor* biases = GetInput(context, node, 2);
            const TfLiteTensor* output = GetOutput(context, node, 0);

            auto* user_data = reinterpret_cast<UserData*>(node->user_data);

            // set param values not parsed from custom options
            user_data->options.K_h = weights->dims->data[1];
            user_data->options.K_w = weights->dims->data[2];
            user_data->options.C_in = weights->dims->data[3] * weights->dims->data[5];
            user_data->options.C_out = weights->dims->data[0] * weights->dims->data[4];

            if (user_data->params->blocks)
            {
                // need to deinit first because something significant has changes with the operator
                conv2d_deepin_deepout_deinit(user_data->params);
            }

            nn_conv2d_init_params_t init_params;
            nn_conv2d_region_params_t  region_params;

            init_params.X_height = input->dims->data[1];
            init_params.X_width = input->dims->data[2];
            init_params.K_h = user_data->options.K_h;
            init_params.K_w = user_data->options.K_w;
            init_params.C_in = user_data->options.C_in;
            init_params.C_out = user_data->options.C_out;
            init_params.pad_mode = user_data->options.padding_mode;
            init_params.zero_point = input->params.zero_point;

            region_params.top = 0;
            region_params.left = 0;
            region_params.rows = output->dims->data[1];
            region_params.cols = output->dims->data[2];

            // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
            //         TfLiteTensor to replace TfLiteQuantizationParams params
            conv2d_deepin_deepout_init(
                user_data->params,
                &init_params,
                &region_params,
                weights->data.int8,
                (data16_t*) biases->data.i16
            );

            return kTfLiteOk;
        }

        TfLiteStatus Eval2D(TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input = GetInput(context, node, 0);
            const TfLiteTensor* weights = GetInput(context, node, 1);
            const TfLiteTensor* shift_scale = GetInput(context, node, 3);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* user_data = reinterpret_cast<UserData*>(node->user_data);

            conv2d_deepin_deepout(
                output->data.int8, // Y
                user_data->params,
                input->data.int8, // X,
                weights->data.int8, // K
                (int16_t*) shift_scale->data.i16 // shifts & scales
            );

            return kTfLiteOk;
        }

    }  // namespace dido

}  // namespace conv


TfLiteRegistration* Register_Conv2D_DIDO() {
    static TfLiteRegistration r = {
        conv::dido::Init2D,
        conv::dido::Free2D,
        conv::dido::Prepare2D,
        conv::dido::Eval2D
    };
    return &r;
}

TfLiteRegistration* Register_Conv2D_SIDO() {
    static TfLiteRegistration r = {
        conv::sido::Init2D,
        conv::sido::Free2D,
        conv::sido::Prepare2D,
        conv::sido::Eval2D
    };
    return &r;
}


}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

