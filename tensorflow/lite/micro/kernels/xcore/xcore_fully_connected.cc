#include "lib_ops/api/fully_connected.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace fully_connected {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::fully_connected::FullyConnected_16), &data);
  ::xcore::fully_connected::FullyConnected_16* op =
      new (data)::xcore::fully_connected::FullyConnected_16();

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* weights = GetInput(context, node, 1);

  int32_t C_in = weights->dims->data[1];
  int32_t C_out = weights->dims->data[0];

  auto* op = reinterpret_cast<::xcore::fully_connected::FullyConnected_16*>(
      node->user_data);
  op->Init(C_in, C_out);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* weights = GetInput(context, node, 1);
  const TfLiteTensor* bso = GetInput(context, node, 2);

  TfLiteTensor* output = GetOutput(context, node, 0);

  auto* op = reinterpret_cast<::xcore::fully_connected::FullyConnected_16*>(
      node->user_data);
  op->Eval(output->data.i16, weights->data.int8, input->data.int8,
           bso->data.i16);

  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FullyConnected_16() {
  static TfLiteRegistration r = {fully_connected::Init, nullptr,
                                 fully_connected::Prepare,
                                 fully_connected::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
