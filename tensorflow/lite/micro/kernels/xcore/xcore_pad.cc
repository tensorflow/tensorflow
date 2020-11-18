#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/util.h"

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
  padding_values_t pv;
  uint32_t pad_value;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto *op_data = reinterpret_cast<PadOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(PadOpData)));

  // parse custom options
  TFLITE_DCHECK(buffer);
  TFLITE_DCHECK(length > 0);

  op_data->pad_value =
      get_named_uint32_custom_option(context, buffer, length, "pad_values");

  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();
  auto keys = map.Keys();
  auto values = map.Values();

  for (int i = 0; i < map.size(); ++i) {
    const std::string &key = keys[i].AsString().str();
    if (key.compare("padding_values") == 0) {
      // values represent [height, height_offset, width, width_offset]
      const auto &vec = values[i].AsVector();
      op_data->pv.height = vec[0].AsInt32();
      op_data->pv.height_offset = vec[1].AsInt32();
      op_data->pv.width = vec[2].AsInt32();
      op_data->pv.width_offset = vec[3].AsInt32();
      break;
    }
  }

  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);

  // setup runtime parameters
  nn_image_params_t x;
  x.height = (uint32_t)input->dims->data[1];
  x.width = (uint32_t)input->dims->data[2];
  x.channels = (uint32_t)input->dims->data[3];

  size_t type_size;
  GetSizeOfType(context, input->type, &type_size);
  auto bytes_per_pixel = type_size * x.channels;
  TF_LITE_ENSURE(context, bytes_per_pixel % 4 == 0);

  auto *op_data = reinterpret_cast<PadOpData *>(node->user_data);

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
