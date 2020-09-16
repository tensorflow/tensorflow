#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "flatbuffers/flexbuffers.h"

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
  size_t bytes_per_pixel;
  uint32_t pad_value;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  PadOpData *op = nullptr;

  context->AllocatePersistentBuffer(context, sizeof(PadOpData),
                                    reinterpret_cast<void **>(&op));

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  TFLITE_DCHECK(length > 0); 

  op->bytes_per_pixel = (size_t)named_uint32_custom_option(context, buffer, length, "bytes_per_pixel");
  op->pad_value = named_uint32_custom_option(context, buffer, length, "pad_value");

  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();
  auto keys = map.Keys();
  auto values = map.Values();

  for (int i = 0; i < map.size(); ++i) {
    const std::string &key = keys[i].AsString().str();
    if (key.compare("padding_values") == 0) {
      //values represent [height, height_offset, width, width_offset]
      const auto &vec =
            values[i].AsVector();  
        op->pv.height = vec[0].AsInt32();
        op->pv.height_offset = vec[1].AsInt32();
        op->pv.width = vec[2].AsInt32();
        op->pv.width_offset = vec[3].AsInt32();
    }
  }

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  PadOpData *op =
      reinterpret_cast<PadOpData *>(node->user_data);

  // setup runtime parameters
  nn_image_params_t x;
  x.height = (uint32_t)input->dims->data[1];
  x.width = (uint32_t)input->dims->data[2];
  x.channels = (uint32_t)input->dims->data[3];

  pad_prepare(&op->plan, &op->pv, &x, op->bytes_per_pixel);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  PadOpData *op =
      reinterpret_cast<PadOpData *>(node->user_data);

  pad_run((void* )output->data.i32, (void* )input->data.i32, &op->plan, op->pad_value);

  return kTfLiteOk;
}
}  // namespace pad

TfLiteRegistration *Register_Pad() {
  static TfLiteRegistration r = {pad::Init, nullptr,
                                 pad::Prepare, pad::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
