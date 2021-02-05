#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/memory_helpers.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pad {

/*
This is a struct that describes the memory required to configure the operator.
*/
struct PadOpData {
  nn_pad_plan_t plan;
  padding_sizes_t pv;
  uint32_t pad_value;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto *op_data = reinterpret_cast<PadOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(PadOpData)));

  // parse custom options
  TFLITE_DCHECK(buffer);
  TFLITE_DCHECK(length > 0);

  auto parser = CustomOptionParser(buffer, length);
  op_data->pad_value = parser.parseNamedCustomOption("pad_value").AsUInt32();

  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto *input = GetInput(context, node, 0);

  nn_image_params_t x;
  x.height = (uint32_t)input->dims->data[1];
  x.width = (uint32_t)input->dims->data[2];
  x.channels = (uint32_t)input->dims->data[3];

  auto *op_data = reinterpret_cast<PadOpData *>(node->user_data);

  auto *paddings = GetInput(context, node, 1);
  auto *pad_sizes = paddings->data.i32;
  op_data->pv.top = pad_sizes[2];
  op_data->pv.bottom = pad_sizes[3];
  op_data->pv.left = pad_sizes[4];
  op_data->pv.right = pad_sizes[5];

  size_t type_size;
  TF_LITE_ENSURE_OK(context, TfLiteTypeSizeOf(input->type, &type_size));
  auto bytes_per_pixel = type_size * x.channels;
  TF_LITE_ENSURE(context, bytes_per_pixel % 4 == 0);

  pad_prepare(&op_data->plan, &op_data->pv, &x, bytes_per_pixel);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  auto *op_data = reinterpret_cast<PadOpData *>(node->user_data);

  pad_run(output->data.data, input->data.data, &op_data->plan,
          op_data->pad_value);

  return kTfLiteOk;
}
}  // namespace pad

TfLiteRegistration *Register_Pad() {
  static TfLiteRegistration r = {pad::Init, nullptr, pad::Prepare, pad::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
