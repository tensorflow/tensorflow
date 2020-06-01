#include "lib_ops/api/pooling.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pooling {

template <int N, class T>
T unpack(const uint8_t* buffer) {
  T retval = 0;
  for (int i = 0; i < N; ++i) retval |= buffer[i] << (8 * i);
  return retval;
}

//**************************************
//**************************************
//**************************************
// MaxPool
//**************************************
//**************************************
//**************************************
namespace maxpool {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  ::xcore::pooling::PoolingParams pooling_params;
  ::xcore::ExecutionPlan execution_plan;

  if (buffer)
    parse_custom_options(buffer, length, pooling_params, &execution_plan);

  void* data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(::xcore::pooling::MaxPool),
                                    &data);
  ::xcore::pooling::MaxPool* op =
      new (data)::xcore::pooling::MaxPool(pooling_params, execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  auto* op = reinterpret_cast<::xcore::pooling::MaxPool*>(node->user_data);

  op->Init(input->dims->data[1],   // X_h
           input->dims->data[2],   // X_w
           input->dims->data[3],   // C_in
           output->dims->data[1],  // Y_h
           output->dims->data[2],  // Y_w
           output->dims->data[3]   // C_out
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  auto* op = reinterpret_cast<::xcore::pooling::MaxPool*>(node->user_data);

  op->Eval(output->data.int8, input->data.int8);

  return kTfLiteOk;
}

}  // namespace maxpool

//**************************************
//**************************************
//**************************************
// AvgPool
//**************************************
//**************************************
//**************************************
namespace avgpool {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  ::xcore::pooling::PoolingParams pooling_params;
  ::xcore::ExecutionPlan execution_plan;

  if (buffer)
    parse_custom_options(buffer, length, pooling_params, &execution_plan);

  void* data = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(::xcore::pooling::AvgPool),
                                    &data);
  ::xcore::pooling::AvgPool* op =
      new (data)::xcore::pooling::AvgPool(pooling_params, execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  auto* op = reinterpret_cast<::xcore::pooling::AvgPool*>(node->user_data);

  op->Init(input->dims->data[1],   // X_h
           input->dims->data[2],   // X_w
           input->dims->data[3],   // C_in
           output->dims->data[1],  // Y_h
           output->dims->data[2],  // Y_w
           output->dims->data[3]   // C_out
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  auto* op = reinterpret_cast<::xcore::pooling::AvgPool*>(node->user_data);

  op->Eval(output->data.int8, input->data.int8);

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

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  ::xcore::ExecutionPlan execution_plan;

  if (buffer) parse_custom_options(buffer, length, &execution_plan);

  void* data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::pooling::AvgPool_Global), &data);
  ::xcore::pooling::AvgPool_Global* op =
      new (data)::xcore::pooling::AvgPool_Global(execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* bss = GetInput(context, node, 1);

  auto* op =
      reinterpret_cast<::xcore::pooling::AvgPool_Global*>(node->user_data);

  op->Init(unpack<4, int32_t>(&bss->data.uint8[0]),   // bias
           unpack<2, uint32_t>(&bss->data.uint8[5]),  // shift
           unpack<1, uint32_t>(&bss->data.uint8[4])   // scal
  );

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  auto* op =
      reinterpret_cast<::xcore::pooling::AvgPool_Global*>(node->user_data);

  op->Eval(output->data.int8,     // Y
           input->data.int8,      // X
           input->dims->data[1],  // X_h
           input->dims->data[2],  // X_w
           input->dims->data[3]   // C_in
  );
  return kTfLiteOk;
}

}  // namespace avgpool_global

}  // namespace pooling

TfLiteRegistration* Register_MaxPool2D() {
  static TfLiteRegistration r = {pooling::maxpool::Init, nullptr,
                                 pooling::maxpool::Prepare,
                                 pooling::maxpool::Eval};
  return &r;
}

TfLiteRegistration* Register_AvgPool2D() {
  static TfLiteRegistration r = {pooling::avgpool::Init, nullptr,
                                 pooling::avgpool::Prepare,
                                 pooling::avgpool::Eval};
  return &r;
}

TfLiteRegistration* Register_AvgPool2D_Global() {
  static TfLiteRegistration r = {pooling::avgpool_global::Init, nullptr,
                                 pooling::avgpool_global::Prepare,
                                 pooling::avgpool_global::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
