#include <iostream>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include "lib_ops/api/conv2d.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

static void parse_options(const char *buffer, size_t length,
                          ::xcore::conv::Conv2DOptions *options,
                          ::xcore::ParPlan *par = nullptr,
                          ::xcore::conv::Conv2DUnpaddedShape *us = nullptr) {
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
      auto vec =
          values[i].AsVector();  // values represent [top, left, zero_point]
      options->padding.data.top = vec[0].AsInt32();
      options->padding.data.left = vec[1].AsInt32();
      options->padding.data.zero_point = vec[2].AsInt32();
    } else if (key.compare("unpadded_shape") == 0) {
      assert(us);
      auto vec =
          values[i].AsVector();  // values represent [C_out, K_h, K_w, C_in]
      us->C_out = vec[0].AsInt32();
      options->K_h = vec[1].AsInt32();
      options->K_w = vec[2].AsInt32();
      us->C_in = vec[3].AsInt32();
    } else if (key.compare("stride_h") == 0)
      options->stride_h = values[i].AsInt32();
    else if (key.compare("stride_w") == 0)
      options->stride_w = values[i].AsInt32();
    else if (key.compare("stride") == 0) {
      auto vec = values[i].AsVector();  // values represent [stride_h, stride_w]
      options->stride_h = vec[0].AsInt32();
      options->stride_w = vec[1].AsInt32();
    } else if (key.compare("par_plan") == 0) {
      assert(par);
      auto jobs = values[i].AsVector();
      for (int i = 0; i < jobs.size(); ++i) {
        auto region = jobs[i].AsVector();
        par->emplace_back(region[0].AsInt32(), region[1].AsInt32(),
                          region[2].AsInt32(), region[3].AsInt32());
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

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2D_SIDO *op = new ::xcore::conv::Conv2D_SIDO();

  if (buffer)
    parse_options(buffer, length, &op->options, nullptr, &op->unpadded_shape);
  return op;
}

void Free(TfLiteContext *context, void *buffer) {
  auto *op = reinterpret_cast<::xcore::conv::Conv2D_SIDO *>(buffer);
  delete op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bias = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_SIDO *>(node->user_data);

  // op->options.C_in = input->dims->data[3];  // number of channels after
  // padding

  op->Init(input->dims->data[1],      // X_h
           input->dims->data[2],      // X_w
           input->dims->data[3],      // C_in (after padding)
           output->dims->data[1],     // Y_h
           output->dims->data[2],     // Y_w
           input->params.zero_point,  // zero_point
           weights->data.int8, bias->data.i16);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *shift_scale = GetInput(context, node, 3);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_SIDO *>(node->user_data);

  op->Eval(output->data.int8,     // Y
           input->data.int8,      // X,
           weights->data.int8,    // K
           shift_scale->data.i16  // shifts & scales
  );

  return kTfLiteOk;
}

}  // namespace sido

//**************************************
//**************************************
//**************************************
// Conv2D_Deepin_Deepout
//**************************************
//**************************************
//**************************************
namespace dido {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2D_DIDO *op = new ::xcore::conv::Conv2D_DIDO();

  if (buffer) parse_options(buffer, length, &op->options, &op->par);
  return op;
}

void Free(TfLiteContext *context, void *buffer) {
  auto *op = reinterpret_cast<::xcore::conv::Conv2D_DIDO *>(buffer);
  delete op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bias = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_DIDO *>(node->user_data);

  // set param values not parsed from custom options
  op->options.K_h = weights->dims->data[1];
  op->options.K_w = weights->dims->data[2];

  op->Init(input->dims->data[1],                             // X_h
           input->dims->data[2],                             // X_w
           weights->dims->data[3] * weights->dims->data[5],  // C_in
           output->dims->data[1],                            // Y_w
           output->dims->data[2],                            // Y_h
           weights->dims->data[0] * weights->dims->data[4],  // C_out
           input->params.zero_point,                         // zero_point
           weights->data.int8, bias->data.i16);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *shift_scale = GetInput(context, node, 3);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_DIDO *>(node->user_data);
  op->Eval(output->data.int8, input->data.int8, weights->data.int8,
           shift_scale->data.i16);

  return kTfLiteOk;
}

}  // namespace dido

namespace n1x1 {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2D_1x1 *op = new ::xcore::conv::Conv2D_1x1();

  if (buffer) parse_options(buffer, length, &op->options);
  return op;
}

void Free(TfLiteContext *context, void *buffer) {
  auto *op = reinterpret_cast<::xcore::conv::Conv2D_1x1 *>(buffer);
  delete op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_1x1 *>(node->user_data);

  op->Init(input->dims->data[1],   // X_h
           input->dims->data[2],   // X_w
           input->dims->data[3],   // C_in
           output->dims->data[1],  // Y_h
           output->dims->data[2],  // Y_w
           output->dims->data[3],  // C_out
           0, 0,
           output->dims->data[1] * output->dims->data[2]  // out_pixels
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bias_shift_scale = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_1x1 *>(node->user_data);

  op->Eval(output->data.int8,          // Y
           input->data.int8,           // X,
           weights->data.int8,         // K
           bias_shift_scale->data.i16  // BSS
  );

  return kTfLiteOk;
}

}  // namespace n1x1

//**************************************
//**************************************
//**************************************
// depthwise
//**************************************
//**************************************
//**************************************

namespace depthwise {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2D_Depthwise *op = new ::xcore::conv::Conv2D_Depthwise();

  if (buffer) parse_options(buffer, length, &op->options);
  return op;
}

void Free(TfLiteContext *context, void *buffer) {
  auto *op = reinterpret_cast<::xcore::conv::Conv2D_Depthwise *>(buffer);
  delete op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op =
      reinterpret_cast<::xcore::conv::Conv2D_Depthwise *>(node->user_data);

  // set param values not parsed from custom options
  op->options.K_h = weights->dims->data[0];
  op->options.K_w = weights->dims->data[1];

  op->Init(input->dims->data[1],   // X_h
           input->dims->data[2],   // X_w
           input->dims->data[3],   // C_in
           output->dims->data[1],  // Y_h
           output->dims->data[2],  // Y_w
           output->dims->data[3]   // C_out
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bss = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op =
      reinterpret_cast<::xcore::conv::Conv2D_Depthwise *>(node->user_data);

  op->Eval(output->data.int8,   // Y
           input->data.int8,    // X,
           weights->data.int8,  // K
           bss->data.i16        // BSS
  );

  return kTfLiteOk;
}

}  // namespace depthwise

}  // namespace conv

TfLiteRegistration *Register_Conv2D_DIDO() {
  static TfLiteRegistration r = {conv::dido::Init, conv::dido::Free,
                                 conv::dido::Prepare, conv::dido::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_SIDO() {
  static TfLiteRegistration r = {conv::sido::Init, conv::sido::Free,
                                 conv::sido::Prepare, conv::sido::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_1x1() {
  static TfLiteRegistration r = {conv::n1x1::Init, conv::n1x1::Free,
                                 conv::n1x1::Prepare, conv::n1x1::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_depthwise() {
  static TfLiteRegistration r = {conv::depthwise::Init, conv::depthwise::Free,
                                 conv::depthwise::Prepare,
                                 conv::depthwise::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
