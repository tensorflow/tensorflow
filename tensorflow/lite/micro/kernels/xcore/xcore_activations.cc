#include "operators/activations.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace activations {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* data = nullptr;
  context->AllocatePersistentBuffer(
      context, sizeof(::xcore::activations::Lookup8), &data);
  ::xcore::activations::Lookup8* op = new (data)::xcore::activations::Lookup8();

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* lut = GetInput(context, node, 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  int32_t length = input->bytes / sizeof(uint8_t);

  auto* op = reinterpret_cast<::xcore::activations::Lookup8*>(node->user_data);
  op->Eval(output->data.uint8, input->data.uint8, lut->data.uint8, length);

  return kTfLiteOk;
}

}  // namespace activations

TfLiteRegistration* Register_Lookup_8() {
  static TfLiteRegistration r = {activations::Init, nullptr,
                                 activations::Prepare, activations::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
