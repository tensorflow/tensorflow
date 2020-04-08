#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

#include "lib_ops/api/conv2d.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

//**************************************
//**************************************
//**************************************
// Shallowin_Deepout
//**************************************
//**************************************
//**************************************
namespace sido {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2DParams conv2d_legacy_params;
  ::xcore::conv::Conv2DUnpaddedShape unpadded_shape;
  padding_mode_t padding_mode;
  if (buffer)
    parse_custom_options(buffer, length, conv2d_legacy_params, &unpadded_shape,
                         nullptr, &padding_mode);

  void *data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(::xcore::conv::Conv2D_SIDO),
                                    &data);
  ::xcore::conv::Conv2D_SIDO *op = new (data)::xcore::conv::Conv2D_SIDO(
      conv2d_legacy_params, unpadded_shape, padding_mode);

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bias = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_SIDO *>(node->user_data);

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
// Conv2D_Deep
//**************************************
//**************************************
//**************************************
namespace deep {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2DParams conv2d_params;
  ::xcore::ParRegionArray par_regions;

  if (buffer)
    parse_custom_options(buffer, length, conv2d_params, nullptr, &par_regions);

  void *data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(::xcore::conv::Conv2D_Deep),
                                    &data);
  ::xcore::conv::Conv2D_Deep *op =
      new (data)::xcore::conv::Conv2D_Deep(conv2d_params, par_regions);

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_Deep *>(node->user_data);

  // set param values not parsed from custom options
  op->params.K_h = weights->dims->data[1];
  op->params.K_w = weights->dims->data[2];

  op->Init(input->dims->data[1],   // X_h
           input->dims->data[2],   // X_w
           input->dims->data[3],   // C_in
           output->dims->data[1],  // Y_w
           output->dims->data[2],  // Y_h
           output->dims->data[3]   // C_out
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bss = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_Deep *>(node->user_data);
  op->Eval(output->data.int8, input->data.int8, weights->data.int8,
           bss->data.i16);

  return kTfLiteOk;
}

}  // namespace deep

namespace n1x1 {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2DParams conv2d_legacy_params;
  padding_mode_t padding_mode;

  if (buffer)
    parse_custom_options(buffer, length, conv2d_legacy_params, nullptr, nullptr,
                         &padding_mode);

  void *data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(::xcore::conv::Conv2D_1x1),
                                    &data);
  ::xcore::conv::Conv2D_1x1 *op =
      new (data)::xcore::conv::Conv2D_1x1(conv2d_legacy_params, padding_mode);

  return op;
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
  ::xcore::conv::Conv2DParams conv2d_params;
  if (buffer) parse_custom_options(buffer, length, conv2d_params);

  void *data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::conv::Conv2D_Depthwise), &data);
  ::xcore::conv::Conv2D_Depthwise *op =
      new (data)::xcore::conv::Conv2D_Depthwise(conv2d_params);

  return op;
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
  op->params.K_h = weights->dims->data[0];
  op->params.K_w = weights->dims->data[1];

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

TfLiteRegistration *Register_Conv2D_Deep() {
  static TfLiteRegistration r = {conv::deep::Init, nullptr, conv::deep::Prepare,
                                 conv::deep::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_SIDO() {
  static TfLiteRegistration r = {conv::sido::Init, nullptr, conv::sido::Prepare,
                                 conv::sido::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_1x1() {
  static TfLiteRegistration r = {conv::n1x1::Init, nullptr, conv::n1x1::Prepare,
                                 conv::n1x1::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_depthwise() {
  static TfLiteRegistration r = {conv::depthwise::Init, nullptr,
                                 conv::depthwise::Prepare,
                                 conv::depthwise::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
