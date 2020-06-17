#include "lib_ops/api/arg_min_max.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace argmax {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::arg_min_max::ArgMax16), &data);
  ::xcore::arg_min_max::ArgMax16* op =
      new (data)::xcore::arg_min_max::ArgMax16();

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  int32_t length = input->bytes / sizeof(int16_t);

  auto* op = reinterpret_cast<::xcore::arg_min_max::ArgMax16*>(node->user_data);

  op->Eval(output->data.i32, input->data.i16, length);

  return kTfLiteOk;
}

}  // namespace argmax

TfLiteRegistration* Register_ArgMax_16() {
  static TfLiteRegistration r = {argmax::Init, nullptr, argmax::Prepare,
                                 argmax::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
