#include <iostream>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

// extern "C" {
//     #include "lib_nn/api/nn_operator.h"
//     #include "lib_nn/api/nn_types.h"
// }

#include "lib_ops/api/pooling.h"

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

static void parse_options(const char* buffer, size_t length,
                          ::xcore::pooling::PoolingOptions* options) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

  auto keys = map.Keys();
  auto values = map.Values();
  for (int i = 0; i < map.size(); ++i) {
    std::string key(keys[i].ToString());

    if (key.compare("pool") == 0) {
      auto vec = values[i].AsVector();  // values represent [pool_h, pool_w]
      options->pool_h = vec[0].AsInt32();
      options->pool_w = vec[1].AsInt32();
    } else if (key.compare("stride") == 0) {
      auto vec = values[i].AsVector();  // values represent [stride_h, stride_w]
      options->stride_h = vec[0].AsInt32();
      options->stride_w = vec[1].AsInt32();
    }
  }
  std::cout << std::endl;
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
  ::xcore::pooling::MaxPool* op = new ::xcore::pooling::MaxPool();

  if (buffer) parse_options(buffer, length, &op->options);
  return op;
}

void Free(TfLiteContext* context, void* buffer) {
  auto* op = reinterpret_cast<::xcore::pooling::MaxPool*>(buffer);
  delete op;
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
  ::xcore::pooling::AvgPool* op = new ::xcore::pooling::AvgPool();

  if (buffer) parse_options(buffer, length, &op->options);
  return op;
}

void Free(TfLiteContext* context, void* buffer) {
  auto* op = reinterpret_cast<::xcore::pooling::AvgPool*>(buffer);
  delete op;
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
  ::xcore::pooling::AvgPool_Global* op = new ::xcore::pooling::AvgPool_Global();

  return op;
}

void Free(TfLiteContext* context, void* buffer) {
  auto* op = reinterpret_cast<::xcore::pooling::AvgPool_Global*>(buffer);
  delete op;
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
  static TfLiteRegistration r = {pooling::maxpool::Init, pooling::maxpool::Free,
                                 pooling::maxpool::Prepare,
                                 pooling::maxpool::Eval};
  return &r;
}

TfLiteRegistration* Register_AvgPool2D() {
  static TfLiteRegistration r = {pooling::avgpool::Init, pooling::avgpool::Free,
                                 pooling::avgpool::Prepare,
                                 pooling::avgpool::Eval};
  return &r;
}

TfLiteRegistration* Register_AvgPool2D_Global() {
  static TfLiteRegistration r = {
      pooling::avgpool_global::Init, pooling::avgpool_global::Free,
      pooling::avgpool_global::Prepare, pooling::avgpool_global::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
