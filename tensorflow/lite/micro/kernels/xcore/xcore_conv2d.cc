#include "lib_ops/api/conv2d.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

//**************************************
//**************************************
//**************************************
// Shallow
//**************************************
//**************************************
//**************************************
namespace shallow {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2DParams conv2d_params;
  ::xcore::ExecutionPlan execution_plan;

  if (buffer)
    parse_custom_options(buffer, length, conv2d_params, &execution_plan);

  void *data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::conv::Conv2D_Shallow), &data);
  ::xcore::conv::Conv2D_Shallow *op =
      new (data)::xcore::conv::Conv2D_Shallow(conv2d_params, execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_Shallow *>(node->user_data);

  // set param values not parsed from custom options
  op->params.K_h = weights->dims->data[1];

  op->Prepare(input->dims->data[1],    // X_h
              input->dims->data[2],    // X_w
              input->dims->data[3],    // C_in
              output->dims->data[1],   // Y_h
              output->dims->data[2],   // Y_w
              weights->dims->data[0],  // C_out
              weights->dims->data[2],  // K_w padded
              weights->data.int8,      // K
              bso->data.i16            // BSO
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_Shallow *>(node->user_data);

  op->Eval(output->data.int8,  // Y
           input->data.int8    // X
  );

  return kTfLiteOk;
}

}  // namespace shallow

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
  ::xcore::ExecutionPlan execution_plan;

  if (buffer)
    parse_custom_options(buffer, length, conv2d_params, &execution_plan);

  void *data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(::xcore::conv::Conv2D_Deep),
                                    &data);
  ::xcore::conv::Conv2D_Deep *op =
      new (data)::xcore::conv::Conv2D_Deep(conv2d_params, execution_plan);
  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_Deep *>(node->user_data);

  // set param values not parsed from custom options
  op->params.K_h = weights->dims->data[1];
  op->params.K_w = weights->dims->data[2];

  op->Prepare(input->dims->data[1],   // X_h
              input->dims->data[2],   // X_w
              input->dims->data[3],   // C_in
              output->dims->data[1],  // Y_w
              output->dims->data[2],  // Y_h
              output->dims->data[3],  // C_out
              weights->data.int8,     // K
              bso->data.i16           // BSO
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_Deep *>(node->user_data);
  op->Eval(output->data.int8, input->data.int8);

  return kTfLiteOk;
}

}  // namespace deep

//**************************************
//**************************************
//**************************************
// 1x1
//**************************************
//**************************************
//**************************************
namespace n1x1 {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  ::xcore::conv::Conv2DParams conv2d_params;
  ::xcore::ExecutionPlan execution_plan;

  if (buffer)
    parse_custom_options(buffer, length, conv2d_params, &execution_plan);

  void *data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(::xcore::conv::Conv2D_1x1),
                                    &data);
  ::xcore::conv::Conv2D_1x1 *op =
      new (data)::xcore::conv::Conv2D_1x1(conv2d_params, execution_plan);

  return data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_1x1 *>(node->user_data);

  op->Prepare(input->dims->data[1],   // X_h
              input->dims->data[2],   // X_w
              input->dims->data[3],   // C_in
              output->dims->data[1],  // Y_h
              output->dims->data[2],  // Y_w
              output->dims->data[3],  // C_out
              weights->data.int8,     // K
              bso->data.i16           // BSO
  );
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op = reinterpret_cast<::xcore::conv::Conv2D_1x1 *>(node->user_data);

  op->Eval(output->data.int8,  // Y
           input->data.int8    // X
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
  ::xcore::ExecutionPlan execution_plan;

  if (buffer)
    parse_custom_options(buffer, length, conv2d_params, &execution_plan);

  void *data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::conv::Conv2D_Depthwise), &data);
  ::xcore::conv::Conv2D_Depthwise *op =
      new (data)::xcore::conv::Conv2D_Depthwise(conv2d_params, execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op =
      reinterpret_cast<::xcore::conv::Conv2D_Depthwise *>(node->user_data);

  // set param values not parsed from custom options
  op->params.K_h = weights->dims->data[0];
  op->params.K_w = weights->dims->data[1];

  op->Prepare(input->dims->data[1],   // X_h
              input->dims->data[2],   // X_w
              input->dims->data[3],   // C_in
              output->dims->data[1],  // Y_h
              output->dims->data[2],  // Y_w
              output->dims->data[3],  // C_out
              weights->data.int8,     // K
              bso->data.i16           // B
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op =
      reinterpret_cast<::xcore::conv::Conv2D_Depthwise *>(node->user_data);

  op->Eval(output->data.int8,  // Y
           input->data.int8    // X
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

TfLiteRegistration *Register_Conv2D_Shallow() {
  static TfLiteRegistration r = {conv::shallow::Init, nullptr,
                                 conv::shallow::Prepare, conv::shallow::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_1x1() {
  static TfLiteRegistration r = {conv::n1x1::Init, nullptr, conv::n1x1::Prepare,
                                 conv::n1x1::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_Depthwise() {
  static TfLiteRegistration r = {conv::depthwise::Init, nullptr,
                                 conv::depthwise::Prepare,
                                 conv::depthwise::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
