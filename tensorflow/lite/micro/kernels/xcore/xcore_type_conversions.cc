#include <iostream>

#include "lib_ops/api/type_conversions.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace type_conversions {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  ::xcore::ExecutionPlan execution_plan;
  if (buffer) parse_custom_options(buffer, length, &execution_plan);

  void* data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::type_conversions::Requantize_16_to_8), &data);
  ::xcore::type_conversions::Requantize_16_to_8* op =
      new (data)::xcore::type_conversions::Requantize_16_to_8(execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  int32_t length = input->bytes / sizeof(int16_t);

  auto* op = reinterpret_cast<::xcore::type_conversions::Requantize_16_to_8*>(
      node->user_data);

  op->Init(length);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  int32_t length = input->bytes / sizeof(int16_t);

  auto* op = reinterpret_cast<::xcore::type_conversions::Requantize_16_to_8*>(
      node->user_data);

  op->Eval(output->data.int8, input->data.i16);

  return kTfLiteOk;
}

}  // namespace type_conversions

TfLiteRegistration* Register_Requantize_16_to_8() {
  static TfLiteRegistration r = {type_conversions::Init, nullptr,
                                 type_conversions::Prepare,
                                 type_conversions::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
