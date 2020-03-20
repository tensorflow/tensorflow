#include <iostream>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "flatbuffers/flexbuffers.h"

#include "lib_ops/api/conv2d.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

    // static void parse_options(
    //     const char* buffer, size_t length, 
    //     ::xcore::conv::Conv2DOptions* options,
    //     ::xcore::ParPlan* par=nullptr)
    // {
    //     const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
    //     // std::cout << flexbuffers::GetRoot(buffer_t, length).ToString() << std::endl;
    //     auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

    //     auto keys = map.Keys();
    //     auto values = map.Values();
    //     for (int i = 0; i < map.size(); ++i)
    //     {
    //         std::string key(keys[i].ToString());
            
    //         if (key.compare("padding") == 0)
    //         {
    //             std::string padding_mode_str = values[i].ToString();
    //             if (padding_mode_str.compare("VALID") == 0) 
    //                 options->padding_mode = PADDING_VALID;
    //             else 
    //                 options->padding_mode = PADDING_SAME;
    //         }
    //         else if (key.compare("unpadded_shape") == 0)
    //         {
    //             auto vec = values[i].AsVector(); // values represent [C_out, K_h, K_w, C_in]
    //             options->C_out = vec[0].AsInt32();
    //             options->K_h = vec[1].AsInt32();
    //             options->K_w = vec[2].AsInt32();
    //             options->C_in = vec[3].AsInt32();
    //         }
    //         else if (key.compare("stride_h") == 0)
    //             options->stride_h = values[i].AsInt32();
    //         else if (key.compare("stride_w") == 0)
    //             options->stride_w = values[i].AsInt32();
    //         else if (key.compare("par_plan") == 0)
    //         {
    //             auto jobs = values[i].AsVector();
    //             for (int i=0; i<jobs.size(); ++i)
    //             {
    //                 auto region = jobs[i].AsVector();
    //                 par->emplace_back(region[0].AsInt32(), region[1].AsInt32(), region[2].AsInt32(), region[3].AsInt32());
    //             }
    //         }
    //     }
union padding_t {
  padding_mode_t mode;
  struct padding_data_t {
    int8_t top;
    int8_t left;
    int8_t zero_point;
  } data;
};

typedef struct {
  padding_t padding;
  int32_t C_in;
  int32_t C_out;
  int32_t K_h;
  int32_t K_w;
  int32_t stride_h;
  int32_t stride_w;
  ParPlan par_plan;
} Conv2DOptions;

static void parse_options(const char *buffer, size_t length,
                          Conv2DOptions *options) {
  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);
  // std::cout << flexbuffers::GetRoot(buffer_t, length).ToString() <<
  // std::endl;
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

  auto keys = map.Keys();
  auto values = map.Values();
  for (int i = 0; i < map.size(); ++i) {
    std::string key(keys[i].ToString());

    if (key.compare("padding") == 0) {
      std::string padding_mode_str = values[i].ToString();
      if (padding_mode_str.compare("VALID") == 0)
          options->padding.mode = PADDING_VALID;
      else
          options->padding.mode = PADDING_SAME;
    } else if (key.compare("pad") == 0) {
      auto vec = values[i].AsVector(); // values represent [top, left, zero_point]
      options->padding.data.top = vec[0].AsInt32();
      options->padding.data.left = vec[1].AsInt32();
      options->padding.data.zero_point = vec[2].AsInt32();
    } else if (key.compare("unpadded_shape") == 0) {
      auto vec = values[i].AsVector(); // values represent [C_out, K_h, K_w, C_in]
      options->C_out = vec[0].AsInt32();
      options->K_h = vec[1].AsInt32();
      options->K_w = vec[2].AsInt32();
      options->C_in = vec[3].AsInt32();
    } else if (key.compare("stride_h") == 0)
      options->stride_h = values[i].AsInt32();
    else if (key.compare("stride_w") == 0)
      options->stride_w = values[i].AsInt32();
    else if (key.compare("stride") == 0) {
      auto vec = values[i].AsVector(); // values represent [stride_h, stride_w]
      options->stride_h = vec[0].AsInt32();
      options->stride_w = vec[1].AsInt32();
    } else if (key.compare("par_plan") == 0) {
      auto jobs = values[i].AsVector();
      options->par_plan.n_regions = jobs.size();
      for (int i = 0; i < jobs.size(); ++i) {
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
} OpData;

void *Init2D(TfLiteContext *context, const char *buffer, size_t length) {
  OpData *op_data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(OpData), (void **)&op_data);

  op_data->params = new nn_conv2d_sido_params_t();

  if (buffer)
    parse_options(buffer, length, &op_data->options);

  return op_data;
}

void Free2D(TfLiteContext *context, void *buffer) {
  auto *op_data = reinterpret_cast<OpData *>(buffer);
  conv2d_shallowin_deepout_deinit(op_data->params);
}

TfLiteStatus Prepare2D(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *biases = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op_data = reinterpret_cast<OpData *>(node->user_data);

  if (op_data->params->blocks) {
    // need to deinit first because something significant has changes with the
    // operator
    conv2d_shallowin_deepout_deinit(op_data->params);
  }

  nn_conv2d_init_params_t init_params;
  nn_conv2d_region_params_t region_params;

  init_params.X_height = input->dims->data[1];
  init_params.X_width = input->dims->data[2];
  init_params.K_h = op_data->options.K_h;
  init_params.K_w = op_data->options.K_w;
  init_params.C_in = input->dims->data[3]; // number of channels after padding
  init_params.C_out = op_data->options.C_out;
  init_params.pad_mode = op_data->options.padding.mode;
  init_params.zero_point = input->params.zero_point;

  region_params.top = 0;
  region_params.left = 0;
  region_params.rows = output->dims->data[1];
  region_params.cols = output->dims->data[2];

  // NOTE: There is a comment in common.h about TfLiteQuantization quantization
  // being added to
  //         TfLiteTensor to replace TfLiteQuantizationParams params
  conv2d_shallowin_deepout_init(op_data->params, &init_params, &region_params,
                                weights->data.int8,
                                (data16_t *)biases->data.i16);

  return kTfLiteOk;
}

TfLiteStatus Eval2D(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *shift_scale = GetInput(context, node, 3);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op_data = reinterpret_cast<OpData *>(node->user_data);

  conv2d_shallowin_deepout(output->data.int8, // Y
                           op_data->params,
                           input->data.int8,                // X,
                           weights->data.int8,              // K
                           (int16_t *)shift_scale->data.i16 // shifts & scales
  );

  return kTfLiteOk;
}

} // namespace sido

// //**************************************
// //**************************************
// //**************************************
// // Conv2D_Deepin_Deepout
// //**************************************
// //**************************************
// //**************************************
// namespace dido {

// typedef struct {
//   Conv2DOptions options;
//   nn_conv2d_dido_params_t *params;
// } OpData;

// void *Init2D(TfLiteContext *context, const char *buffer, size_t length) {
//   OpData *op_data = nullptr;
//   context->AllocatePersistentBuffer(context, sizeof(OpData), (void **)&op_data);

//   op_data->params = new nn_conv2d_dido_params_t();

//   if (buffer)
//     parse_options(buffer, length, &op_data->options);

//   return op_data;
// }

// void Free2D(TfLiteContext *context, void *buffer) {
//   auto *op_data = reinterpret_cast<OpData *>(buffer);

//   conv2d_deepin_deepout_deinit(op_data->params);
// }

// TfLiteStatus Prepare2D(TfLiteContext *context, TfLiteNode *node) {
//   TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
//   TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

//   const TfLiteTensor *input = GetInput(context, node, 0);
//   const TfLiteTensor *weights = GetInput(context, node, 1);
//   const TfLiteTensor *bias_shift_scale = GetInput(context, node, 2);
//   const TfLiteTensor *output = GetOutput(context, node, 0);

//   auto *op_data = reinterpret_cast<OpData *>(node->user_data);

//   // set param values not parsed from custom options
//   op_data->options.K_h = weights->dims->data[1];
//   op_data->options.K_w = weights->dims->data[2];
//   op_data->options.C_in = weights->dims->data[3] * weights->dims->data[5];
//   op_data->options.C_out = weights->dims->data[0] * weights->dims->data[4];

//   if (op_data->params->blocks) {
//     // need to deinit first because something significant has changes with the
//     // operator
//     conv2d_deepin_deepout_deinit(op_data->params);
//   }

//   nn_conv2d_init_params_t init_params;
//   nn_conv2d_region_params_t region_params;

//   init_params.X_height = input->dims->data[1];
//   init_params.X_width = input->dims->data[2];
//   init_params.K_h = op_data->options.K_h;
//   init_params.K_w = op_data->options.K_w;
//   init_params.C_in = op_data->options.C_in;
//   init_params.C_out = op_data->options.C_out;
//   init_params.pad_mode = op_data->options.padding.mode;
//   init_params.zero_point = input->params.zero_point;

//   region_params.top = 0;
//   region_params.left = 0;
//   region_params.rows = output->dims->data[1];
//   region_params.cols = output->dims->data[2];

//   // NOTE: There is a comment in common.h about TfLiteQuantization quantization
//   // being added to
//   //         TfLiteTensor to replace TfLiteQuantizationParams params
//   conv2d_deepin_deepout_init(op_data->params, &init_params, &region_params,
//                              weights->data.int8,
//                              (data16_t *)bias_shift_scale->data.i16);

//   return kTfLiteOk;
// }

// TfLiteStatus Eval2D(TfLiteContext *context, TfLiteNode *node) {
//   const TfLiteTensor *input = GetInput(context, node, 0);
//   const TfLiteTensor *weights = GetInput(context, node, 1);
//   const TfLiteTensor *shift_scale = GetInput(context, node, 3);
//   TfLiteTensor *output = GetOutput(context, node, 0);

//   auto *op_data = reinterpret_cast<OpData *>(node->user_data);

//   conv2d_deepin_deepout(output->data.int8, // Y
//                         op_data->params,
//                         input->data.int8,                // X,
//                         weights->data.int8,              // K
//                         (int16_t *)shift_scale->data.i16 // shifts & scales
//   );

//   return kTfLiteOk;
// }

// } // namespace dido

//**************************************
//**************************************
//**************************************
// 1x1
//**************************************
//**************************************
//**************************************

    // //**************************************
    // //**************************************
    // //**************************************
    // // Shallowin_Deepout
    // //**************************************
    // //**************************************
    // //**************************************
    // namespace sido {

    //     typedef struct {
    //         ::xcore::conv::Conv2DOptions options;
    //         nn_conv2d_sido_params_t *params;
    //     } OpData;

    //     void* Init(TfLiteContext* context, const char* buffer, size_t length) 
    //     {
    //         OpData* op_data = nullptr;
    //         context->AllocatePersistentBuffer(context, sizeof(OpData), (void**) &op_data);

    //         op_data->params = new nn_conv2d_sido_params_t();

    //         if (buffer)
    //             parse_options(buffer, length, &op_data->options);

    //         return op_data;
    //     }

    //     void Free(TfLiteContext* context, void* buffer) {
    //         auto* op_data = reinterpret_cast<OpData*>(buffer);
    //         conv2d_shallowin_deepout_deinit(op_data->params);
    //     }

    //     TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    //         TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
    //         TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    //         const TfLiteTensor* input = GetInput(context, node, 0);
    //         const TfLiteTensor* weights = GetInput(context, node, 1);
    //         const TfLiteTensor* biases = GetInput(context, node, 2);
    //         const TfLiteTensor* output = GetOutput(context, node, 0);

    //         auto* op_data = reinterpret_cast<OpData*>(node->user_data);

    //         if (op_data->params->blocks)
    //         {
    //             // need to deinit first because something significant has changes with the operator
    //             conv2d_shallowin_deepout_deinit(op_data->params);
    //         }

    //         nn_conv2d_init_params_t init_params;
    //         nn_conv2d_region_params_t region_params;

    //         init_params.X_height = input->dims->data[1];
    //         init_params.X_width = input->dims->data[2];
    //         init_params.K_h = op_data->options.K_h;
    //         init_params.K_w = op_data->options.K_w;
    //         init_params.C_in = input->dims->data[3]; // number of channels after padding
    //         init_params.C_out = op_data->options.C_out;
    //         init_params.pad_mode = op_data->options.padding_mode;
    //         init_params.zero_point = input->params.zero_point;

    //         region_params.top = 0;
    //         region_params.left = 0;
    //         region_params.rows = output->dims->data[1];
    //         region_params.cols = output->dims->data[2];

    //         // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
    //         //         TfLiteTensor to replace TfLiteQuantizationParams params
    //         conv2d_shallowin_deepout_init(
    //             op_data->params,
    //             &init_params,
    //             &region_params,
    //             weights->data.int8,
    //             (data16_t*) biases->data.i16
    //         );

    //         return kTfLiteOk;
    //     }


    //     TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    //         const TfLiteTensor* input = GetInput(context, node, 0);
    //         const TfLiteTensor* weights = GetInput(context, node, 1);
    //         const TfLiteTensor* shift_scale = GetInput(context, node, 3);
    //         TfLiteTensor* output = GetOutput(context, node, 0);

    //         auto* op_data = reinterpret_cast<OpData*>(node->user_data);

    //         conv2d_shallowin_deepout(
    //             output->data.int8, // Y
    //             op_data->params,
    //             input->data.int8, // X,
    //             weights->data.int8, // K
    //             (int16_t*) shift_scale->data.i16 // shifts & scales
    //         );

    //         return kTfLiteOk;
    //     }

    // } //namespace sido


    //**************************************
    //**************************************
    //**************************************
    // Conv2D_Deepin_Deepout
    //**************************************
    //**************************************
    //**************************************
    namespace dido {

        void* Init(TfLiteContext* context, const char* buffer, size_t length) 
        {
            ::xcore::conv::Conv2D_DIDO* op = new ::xcore::conv::Conv2D_DIDO();

            if (buffer)
                parse_options(buffer, length, &op->options, &op->par);

            return op;
        }

        void Free(TfLiteContext* context, void* buffer) 
        {
            auto* op = reinterpret_cast<::xcore::conv::Conv2D_DIDO*>(buffer);
            delete op;
        }

        TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
            TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
            TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

            const TfLiteTensor* input = GetInput(context, node, 0);
            const TfLiteTensor* weights = GetInput(context, node, 1);
            const TfLiteTensor* bias = GetInput(context, node, 2);
            const TfLiteTensor* output = GetOutput(context, node, 0);

            auto* op = reinterpret_cast<::xcore::conv::Conv2D_DIDO*>(node->user_data);

            // set param values not parsed from custom options
            op->options.K_h = weights->dims->data[1];
            op->options.K_w = weights->dims->data[2];
            op->options.C_in = weights->dims->data[3] * weights->dims->data[5];
            op->options.C_out = weights->dims->data[0] * weights->dims->data[4];

            int32_t X_h = input->dims->data[1];
            int32_t X_w = input->dims->data[2];
            // NOTE: There is a comment in common.h about TfLiteQuantization quantization being added to 
            //         TfLiteTensor to replace TfLiteQuantizationParams params
            int32_t zero_point = input->params.zero_point;
            int32_t rows = output->dims->data[1];
            int32_t cols = output->dims->data[2];
            op->Init(X_h, X_w, zero_point, rows, cols, weights->data.int8, bias->data.i16);

            return kTfLiteOk;
        }

        TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
            const TfLiteTensor* input = GetInput(context, node, 0);
            const TfLiteTensor* weights = GetInput(context, node, 1);
            const TfLiteTensor* shift_scale = GetInput(context, node, 3);
            TfLiteTensor* output = GetOutput(context, node, 0);

            auto* op = reinterpret_cast<::xcore::conv::Conv2D_DIDO*>(node->user_data);
            op->Eval(output->data.int8,
                    input->data.int8,
                    weights->data.int8,
                    shift_scale->data.i16
            );

            return kTfLiteOk;
        }

    }  // namespace dido

    // //**************************************
    // //**************************************
    // //**************************************
    // // 1x1
    // //**************************************
    // //**************************************
    // //**************************************

    // namespace n1x1 {

    //     typedef struct {
    //         ::xcore::conv::Conv2DOptions options;
    //         nn_conv2d_1x1_plan_t plan;
    //     } OpData;

    //     void* Init(TfLiteContext* context, const char* buffer, size_t length)
    //     {
    //         OpData* op_data = nullptr;
    //         context->AllocatePersistentBuffer(context, sizeof(OpData), (void**) &op_data);

    //         if (buffer)
    //             parse_options(buffer, length, &op_data->options);

    //         return op_data;
    //     }

    //     TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    //         TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
    //         TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    //         const TfLiteTensor* input = GetInput(context, node, 0);
    //         const TfLiteTensor* output = GetOutput(context, node, 0);

    //         auto* op_data = reinterpret_cast<OpData*>(node->user_data);

    //         // set param values not parsed from custom options
    //         op_data->options.C_in = input->dims->data[3];
    //         op_data->options.C_out = output->dims->data[3];

    //         nn_image_params_t params_in;
    //         params_in.height = input->dims->data[1];
    //         params_in.width = input->dims->data[2];
    //         params_in.channels = op_data->options.C_in;

    //         nn_image_params_t params_out;
    //         params_out.height = output->dims->data[1];
    //         params_out.width = output->dims->data[2];
    //         params_out.channels = op_data->options.C_out;

    //         conv2d_1x1_init(
    //             &op_data->plan,
    //             &params_in,
    //             &params_out,
    //             0, // start_row
    //             0, // start_col
    //             params_out.height * params_out.width //out_pixels
    //         );

    //         return kTfLiteOk;
    //     }


    //     TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    //         const TfLiteTensor* input = GetInput(context, node, 0);
    //         const TfLiteTensor* weights = GetInput(context, node, 1);
    //         const TfLiteTensor* bias_shift_scale = GetInput(context, node, 2);
    //         TfLiteTensor* output = GetOutput(context, node, 0);

    //         auto* op_data = reinterpret_cast<OpData*>(node->user_data);

    //         conv2d_1x1(
    //             output->data.int8, // Y
    //             input->data.int8, // X,
    //             weights->data.int8, // K
    //             (data16_t*) bias_shift_scale->data.i16, // BSS
    //             &op_data->plan
    //         );

    //         return kTfLiteOk;
    //     }

    // } //namespace n1x1

//}  // namespace conv


// TfLiteRegistration* Register_Conv2D_DIDO() {
//     static TfLiteRegistration r = {
//         conv::dido::Init,
//         conv::dido::Free,
//         conv::dido::Prepare,
//         conv::dido::Eval
//     };
//     return &r;
// }

namespace n1x1 {

  typedef struct {
    Conv2DOptions options;
    nn_conv2d_1x1_plan_t plan;
  } OpData;

  void *Init2D(TfLiteContext *context, const char *buffer, size_t length) {
    OpData *op_data = nullptr;
    context->AllocatePersistentBuffer(context, sizeof(OpData), (void **)&op_data);

    if (buffer)
      parse_options(buffer, length, &op_data->options);

    return op_data;
  }

  TfLiteStatus Prepare2D(TfLiteContext *context, TfLiteNode *node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    const TfLiteTensor *input = GetInput(context, node, 0);
    const TfLiteTensor *output = GetOutput(context, node, 0);

    auto *op_data = reinterpret_cast<OpData *>(node->user_data);

    // set param values not parsed from custom options
    op_data->options.C_in = input->dims->data[3];
    op_data->options.C_out = output->dims->data[3];

    nn_image_params_t params_in;
    params_in.height = input->dims->data[1];
    params_in.width = input->dims->data[2];
    params_in.channels = op_data->options.C_in;

    nn_image_params_t params_out;
    params_out.height = output->dims->data[1];
    params_out.width = output->dims->data[2];
    params_out.channels = op_data->options.C_out;

    conv2d_1x1_init(&op_data->plan, &params_in, &params_out,
                    0,                                   // start_row
                    0,                                   // start_col
                    params_out.height * params_out.width // out_pixels
    );

    return kTfLiteOk;
  }

  TfLiteStatus Eval2D(TfLiteContext *context, TfLiteNode *node) {
    const TfLiteTensor *input = GetInput(context, node, 0);
    const TfLiteTensor *weights = GetInput(context, node, 1);
    const TfLiteTensor *bias_shift_scale = GetInput(context, node, 2);
    TfLiteTensor *output = GetOutput(context, node, 0);

    auto *op_data = reinterpret_cast<OpData *>(node->user_data);

    conv2d_1x1(output->data.int8,                      // Y
              input->data.int8,                       // X,
              weights->data.int8,                     // K
              (data16_t *)bias_shift_scale->data.i16, // BSS
              &op_data->plan);

    return kTfLiteOk;
  }

} // namespace n1x1

//**************************************
//**************************************
//**************************************
// depthwise
//**************************************
//**************************************
//**************************************

namespace depthwise {

  typedef struct {
    Conv2DOptions options;
    nn_conv2d_depthwise_plan_t plan;
    nn_conv2d_depthwise_job_t job;
  } OpData;

  void *Init(TfLiteContext *context, const char *buffer, size_t length) {
    OpData *op_data = nullptr;
    context->AllocatePersistentBuffer(context, sizeof(OpData), (void **)&op_data);

    if (buffer)
      parse_options(buffer, length, &op_data->options);

    return op_data;
  }

  TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    const TfLiteTensor *input = GetInput(context, node, 0);
    const TfLiteTensor *weights = GetInput(context, node, 1);
    const TfLiteTensor *output = GetOutput(context, node, 0);

    auto *op_data = reinterpret_cast<OpData *>(node->user_data);

    // set param values not parsed from custom options
    op_data->options.C_in = input->dims->data[3];
    op_data->options.C_out = output->dims->data[3];
    op_data->options.K_h = weights->dims->data[0];
    op_data->options.K_w = weights->dims->data[1];

    nn_image_params_t params_in;
    params_in.height = input->dims->data[1];
    params_in.width = input->dims->data[2];
    params_in.channels = op_data->options.C_in;

    nn_image_params_t params_out;
    params_out.height = output->dims->data[1];
    params_out.width = output->dims->data[2];
    params_out.channels = op_data->options.C_out;

    int8_t window_start_row;
    int8_t window_start_col;
    int8_t zero_point;

    window_start_row = -op_data->options.padding.data.top;
    window_start_col = -op_data->options.padding.data.left;
    zero_point = op_data->options.padding.data.zero_point;

    conv2d_depthwise_init(&op_data->plan, &op_data->job, &params_in, &params_out,
                          nullptr, // job_params
                          window_start_row, window_start_col,
                          op_data->options.K_h, op_data->options.K_w,
                          op_data->options.stride_h, op_data->options.stride_w,
                          zero_point,
                          1 // job_count
    );

    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
    const TfLiteTensor *input = GetInput(context, node, 0);
    const TfLiteTensor *weights = GetInput(context, node, 1);
    const TfLiteTensor *bss = GetInput(context, node, 2);
    TfLiteTensor *output = GetOutput(context, node, 0);

    auto *op_data = reinterpret_cast<OpData *>(node->user_data);

    conv2d_depthwise(output->data.int8,               // Y
                    input->data.int8,                // X,
                    weights->data.int8,              // K
                    (nn_bss_block_t *)bss->data.i16, // BSS
                    &op_data->plan, &op_data->job);

    return kTfLiteOk;
  }

} // namespace depthwise

} // namespace conv

TfLiteRegistration *Register_Conv2D_DIDO() {
  static TfLiteRegistration r = {conv::dido::Init2D, conv::dido::Free2D,
                                 conv::dido::Prepare2D, conv::dido::Eval2D};
  return &r;
}

TfLiteRegistration *Register_Conv2D_SIDO() {
  static TfLiteRegistration r = {conv::sido::Init2D, conv::sido::Free2D,
                                 conv::sido::Prepare2D, conv::sido::Eval2D};
  return &r;
}

TfLiteRegistration *Register_Conv2D_1x1() {
  static TfLiteRegistration r = {conv::n1x1::Init2D, nullptr,
                                 conv::n1x1::Prepare2D, conv::n1x1::Eval2D};
  return &r;
}

TfLiteRegistration *Register_Conv2D_depthwise() {
  static TfLiteRegistration r = {conv::depthwise::Init, nullptr,
                                 conv::depthwise::Prepare,
                                 conv::depthwise::Eval};
  return &r;
}

// TfLiteRegistration* Register_Conv2D_SIDO() {
//     static TfLiteRegistration r = {
//         conv::sido::Init,
//         conv::sido::Free,
//         conv::sido::Prepare,
//         conv::sido::Eval
//     };
//     return &r;
// }

// TfLiteRegistration* Register_Conv2D_1x1() {
//     static TfLiteRegistration r = {
//         conv::n1x1::Init,
//         nullptr,
//         conv::n1x1::Prepare,
//         conv::n1x1::Eval
//     };
//     return &r;
// }

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
