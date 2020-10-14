#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pooling {

struct PoolingThreadData {
  int8_t* Y;
  const int8_t* X;
};

struct PoolingThreadParams {
  const nn_image_params_t* x_image;
  const nn_image_params_t* y_image;
  const nn_window_params_t* window;
  nn_window_op_job_params_t job;
};

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

struct MaxPoolOpData {
  PoolingParams params;
  ExecutionPlan execution_plan;
  int stack_scratch_index;
  size_t stack_size;
};

struct MaxPoolThreadData {
  PoolingThreadData data;
  PoolingThreadParams params;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void maxpool_thread_worker(void* context) {
  MaxPoolThreadData* td = (MaxPoolThreadData*)context;
  maxpool2d_ext(td->data.Y, td->data.X, td->params.x_image, td->params.y_image,
                td->params.window, &td->params.job, MAXPOOL2D_FLAG_NONE);
}
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  MaxPoolOpData* op = nullptr;
  op = reinterpret_cast<MaxPoolOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(MaxPoolOpData)));
  op->stack_scratch_index = -1;
  op->stack_size = 0;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op->params,
                       &op->execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MaxPoolOpData* op = reinterpret_cast<MaxPoolOpData*>(node->user_data);

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op->stack_size, maxpool_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  MaxPoolOpData* op = reinterpret_cast<MaxPoolOpData*>(node->user_data);
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(maxpool_thread_worker, stack, op->stack_size);

  // create thread data
  int n_th = op->execution_plan.GetNumThreads();
  MaxPoolThreadData thread_data[n_th];

  // setup params common to all thread workers
  nn_image_params_t in_image = {(uint32_t)input->dims->data[1],
                                (uint32_t)input->dims->data[2],
                                (uint32_t)input->dims->data[3]};
  nn_image_params_t out_image = {(uint32_t)output->dims->data[1],
                                 (uint32_t)output->dims->data[2],
                                 (uint32_t)output->dims->data[3]};
  nn_window_params_t pooling_window = {
      {(uint32_t)op->params.pool_h, (uint32_t)op->params.pool_w},
      {0, 0},
      {op->params.stride_h, op->params.stride_w}};

  // create tasks
  for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
    const RowColRegion& region = op->execution_plan.regions[i_rg];
    thread_data[i_rg].data.Y = (nn_image_t*)output->data.int8;
    thread_data[i_rg].data.X = (const nn_image_t*)input->data.int8;
    thread_data[i_rg].params.x_image = &in_image;
    thread_data[i_rg].params.y_image = &out_image;
    thread_data[i_rg].params.window = &pooling_window;
    thread_data[i_rg].params.job = {
        {region.top, region.left, 0},
        {region.rows, region.cols, output->dims->data[3]}};

    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_rg]));
  }

  // start and wait for tasks to complete
  dispatcher->JoinTasks();

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

struct AvgPoolOpData {
  PoolingParams params;
  ExecutionPlan execution_plan;
  int stack_scratch_index;
  size_t stack_size;
};

struct AvgPoolThreadData {
  PoolingThreadData data;
  PoolingThreadParams params;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void avgpool_thread_worker(void* context) {
  AvgPoolThreadData* td = (AvgPoolThreadData*)context;
  avgpool2d_ext(td->data.Y, td->data.X, td->params.x_image, td->params.y_image,
                td->params.window, &td->params.job, AVGPOOL2D_FLAG_NONE);
}
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  AvgPoolOpData* op = nullptr;
  op = reinterpret_cast<AvgPoolOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(AvgPoolOpData)));
  op->stack_scratch_index = -1;
  op->stack_size = 0;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op->params,
                       &op->execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  AvgPoolOpData* op = reinterpret_cast<AvgPoolOpData*>(node->user_data);

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op->stack_size, avgpool_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  AvgPoolOpData* op = reinterpret_cast<AvgPoolOpData*>(node->user_data);
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(avgpool_thread_worker, stack, op->stack_size);

  // create thread data
  int n_th = op->execution_plan.GetNumThreads();
  AvgPoolThreadData thread_data[n_th];

  // setup params common to all thread workers
  nn_image_params_t in_image = {(uint32_t)input->dims->data[1],
                                (uint32_t)input->dims->data[2],
                                (uint32_t)input->dims->data[3]};
  nn_image_params_t out_image = {(uint32_t)output->dims->data[1],
                                 (uint32_t)output->dims->data[2],
                                 (uint32_t)output->dims->data[3]};
  nn_window_params_t pooling_window = {
      {(uint32_t)op->params.pool_h, (uint32_t)op->params.pool_w},
      {0, 0},
      {op->params.stride_h, op->params.stride_w}};

  // create tasks
  for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
    const RowColRegion& region = op->execution_plan.regions[i_rg];
    thread_data[i_rg].data.Y = (nn_image_t*)output->data.int8;
    thread_data[i_rg].data.X = (const nn_image_t*)input->data.int8;
    thread_data[i_rg].params.x_image = &in_image;
    thread_data[i_rg].params.y_image = &out_image;
    thread_data[i_rg].params.window = &pooling_window;
    thread_data[i_rg].params.job = {
        {region.top, region.left, 0},
        {region.rows, region.cols, output->dims->data[3]}};
    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_rg]));
  }

  // start and wait for tasks to complete
  dispatcher->JoinTasks();

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

struct AvgPoolGlobalOpData {
  ExecutionPlan execution_plan;
  int32_t bias;
  int8_t scale;
  uint16_t shift;
  int stack_scratch_index;
  size_t stack_size;
};

struct AvgPoolGlobalThreadData {
  PoolingThreadData data;
  int32_t bias;
  int8_t scale;
  uint16_t shift;
  int32_t chan_start;
  int32_t chan_count;
  const nn_image_params_t* x_image;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void avgpool_global_thread_worker(void* context) {
  AvgPoolGlobalThreadData* td = (AvgPoolGlobalThreadData*)context;
  avgpool2d_global_ext(td->data.Y, td->data.X, td->bias, td->scale, td->shift,
                       td->x_image, td->chan_start, td->chan_count,
                       AVGPOOL2D_GLOBAL_FLAG_NONE);
}
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  AvgPoolGlobalOpData* op = nullptr;
  op = reinterpret_cast<AvgPoolGlobalOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(AvgPoolGlobalOpData)));
  op->stack_scratch_index = -1;
  op->stack_size = 0;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, &op->execution_plan);

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* bss = GetInput(context, node, 1);

  AvgPoolGlobalOpData* op =
      reinterpret_cast<AvgPoolGlobalOpData*>(node->user_data);

  op->bias = unpack<4, int32_t>(&bss->data.uint8[0]);
  op->shift = unpack<2, uint16_t>(&bss->data.uint8[5]);
  op->scale = unpack<1, int8_t>(&bss->data.uint8[4]);

  TFLITE_DCHECK(op->scale > 0);

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op->stack_size, avgpool_global_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  AvgPoolGlobalOpData* op =
      reinterpret_cast<AvgPoolGlobalOpData*>(node->user_data);
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(avgpool_global_thread_worker, stack,
                              op->stack_size);

  // create thread data
  int i_th = 0;
  int n_th = op->execution_plan.GetNumThreads();
  AvgPoolGlobalThreadData thread_data[n_th];

  // setup params common to all thread workers
  nn_image_params_t in_image = {(uint32_t)input->dims->data[1],
                                (uint32_t)input->dims->data[2],
                                (uint32_t)input->dims->data[3]};

  // create tasks
  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup& changrp = op->execution_plan.changrps[i_cg];
    thread_data[i_th].data.Y = output->data.int8;
    thread_data[i_th].data.X = input->data.int8;
    thread_data[i_th].bias = op->bias;
    thread_data[i_th].shift = op->shift;
    thread_data[i_th].scale = op->scale;
    thread_data[i_th].chan_start = changrp.start;
    thread_data[i_th].chan_count = changrp.size;
    thread_data[i_th].x_image = &in_image;

    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_th]));

    i_th++;
    if (i_th == n_th) {
      // start and wait for tasks to complete
      dispatcher->JoinTasks();
      i_th = 0;
    }
  }
  dispatcher->JoinTasks();  // finish up any added tasks

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
