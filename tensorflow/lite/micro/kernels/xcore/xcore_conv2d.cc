#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv {

struct Conv2DThreadData {
  nn_image_t *Y;
  const nn_image_t *X;
  const nn_tensor_t *K;
  const nn_bso_block_t *BSO;
};

//**************************************
//**************************************
//**************************************
// Shallow
//**************************************
//**************************************
//**************************************
namespace shallow {

struct Conv2DShallowOpData {
  Conv2DParams params;
  ExecutionPlan execution_plan;
  nn_conv2d_shallowin_plan_t plan;
  nn_conv2d_shallowin_job_t *jobs;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

struct Conv2DShallowThreadData {
  const nn_conv2d_shallowin_plan_t *plan;
  nn_conv2d_shallowin_job_t *job;
  Conv2DThreadData data;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_shallow_thread_worker(void *context) {
  Conv2DShallowThreadData *td = (Conv2DShallowThreadData *)context;
  conv2d_shallowin(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
                   td->job);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  Conv2DShallowOpData *op = nullptr;
  op = reinterpret_cast<Conv2DShallowOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(Conv2DShallowOpData)));
  op->jobs = nullptr;
  op->stack_scratch_index = -1;
  op->stack_size = 0;
  op->weights_scratch_index = -1;
  op->bias_scratch_index = -1;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op->params,
                       &op->execution_plan);

  // allocate the jobs
  op->jobs = reinterpret_cast<nn_conv2d_shallowin_job_t *>(
      context->AllocatePersistentBuffer(
          context, sizeof(nn_conv2d_shallowin_job_t) *
                       op->execution_plan.changrps.GetSize() *
                       op->execution_plan.regions.GetSize()));

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DShallowOpData *op =
      reinterpret_cast<Conv2DShallowOpData *>(node->user_data);

  // set param values not parsed from custom options
  op->params.K_h = weights->dims->data[1];

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)input->dims->data[1],
                                 (uint32_t)input->dims->data[2],
                                 (uint32_t)input->dims->data[3]};
  nn_image_params_t out_params = {(uint32_t)output->dims->data[1],
                                  (uint32_t)output->dims->data[2],
                                  (uint32_t)weights->dims->data[0]};
  nn_window_params_t conv_window = {
      {(uint32_t)op->params.K_h, (uint32_t)op->params.K_w},
      {-op->params.pad.top, -op->params.pad.left},
      {op->params.stride_h, op->params.stride_w}};

  int32_t n_jobs = op->execution_plan.changrps.GetSize() *
                   op->execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(op->stack_size, conv2d_shallow_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(weights->data.int8)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetWeightsScratchSize(),
        &op->weights_scratch_index));
  }
  if (IS_NOT_RAM(bso->data.i16)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetBiasScratchSize(),
        &op->bias_scratch_index));
  }

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];
      job_params[i_cg * op->execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {region.rows, region.cols, changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_shallowin_init(&op->plan, op->jobs, &in_params, &out_params,
                        &job_params[0], &conv_window, op->params.pad.zero_point,
                        n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DShallowOpData *op =
      reinterpret_cast<Conv2DShallowOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_shallow_thread_worker, stack,
                              op->stack_size);

  // create thread data and tasks
  Conv2DShallowThreadData thread_data[op->execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(&tK, weights->data.int8,
                             op->execution_plan.GetWeightsScratchSize(),
                             changrp);
    dispatcher->FetchBiases(&tBSO, bso->data.i16,
                            op->execution_plan.GetBiasScratchSize(), changrp);

    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * op->execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].data.X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &op->plan;
      op->jobs[i_job].stride.start.K = 0;
      op->jobs[i_job].stride.start.BSO = 0;
      thread_data[i_rg].job = &op->jobs[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace shallow

//**************************************
//**************************************
//**************************************
// Conv2D_Deep
//**************************************
//**************************************
//**************************************
namespace deep {

struct Conv2DDeepOpData {
  Conv2DParams params;
  ExecutionPlan execution_plan;
  nn_conv2d_deep_plan_t plan;
  nn_conv2d_deep_job_t *jobs;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

struct Conv2DDeepThreadData {
  Conv2DThreadData data;
  const nn_conv2d_deep_plan_t *plan;
  const nn_conv2d_deep_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_deep_thread_worker(void *context) {
  Conv2DDeepThreadData *td = (Conv2DDeepThreadData *)context;
  conv2d_deep(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
              td->job);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  Conv2DDeepOpData *op = nullptr;
  op = reinterpret_cast<Conv2DDeepOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(Conv2DDeepOpData)));
  op->jobs = nullptr;
  op->stack_scratch_index = -1;
  op->stack_size = 0;
  op->weights_scratch_index = -1;
  op->bias_scratch_index = -1;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op->params,
                       &op->execution_plan);

  // allocate the jobs
  op->jobs = reinterpret_cast<nn_conv2d_deep_job_t *>(
      context->AllocatePersistentBuffer(
          context, sizeof(nn_conv2d_deep_job_t) *
                       op->execution_plan.changrps.GetSize() *
                       op->execution_plan.regions.GetSize()));

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DDeepOpData *op = reinterpret_cast<Conv2DDeepOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // set param values not parsed from custom options
  op->params.K_h = weights->dims->data[1];
  op->params.K_w = weights->dims->data[2];

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)input->dims->data[1],
                                 (uint32_t)input->dims->data[2],
                                 (uint32_t)input->dims->data[3]};
  nn_image_params_t out_params = {(uint32_t)output->dims->data[1],
                                  (uint32_t)output->dims->data[2],
                                  (uint32_t)output->dims->data[3]};
  nn_window_params_t conv_window = {
      {(uint32_t)op->params.K_h, (uint32_t)op->params.K_w},
      {-op->params.pad.top, -op->params.pad.left},
      {op->params.stride_h, op->params.stride_w}};

  int32_t n_jobs = op->execution_plan.changrps.GetSize() *
                   op->execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(op->stack_size, conv2d_deep_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(weights->data.int8)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetWeightsScratchSize(),
        &op->weights_scratch_index));
  }
  if (IS_NOT_RAM(bso->data.i16)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetBiasScratchSize(),
        &op->bias_scratch_index));
  }

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];
      job_params[i_cg * op->execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {region.rows, region.cols, changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_deep_init(&op->plan, op->jobs, &in_params, &out_params, &job_params[0],
                   &conv_window, op->params.pad.zero_point, n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DDeepOpData *op = reinterpret_cast<Conv2DDeepOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_deep_thread_worker, stack, op->stack_size);

  // create thread data and tasks
  Conv2DDeepThreadData thread_data[op->execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(&tK, weights->data.int8,
                             op->execution_plan.GetWeightsScratchSize(),
                             changrp);
    dispatcher->FetchBiases(&tBSO, bso->data.i16,
                            op->execution_plan.GetBiasScratchSize(), changrp);

    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * op->execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].data.X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &op->plan;
      op->jobs[i_job].stride.start.K = 0;
      op->jobs[i_job].stride.start.BSO = 0;
      thread_data[i_rg].job = &op->jobs[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace deep

//**************************************
//**************************************
//**************************************
// 1x1
//**************************************
//**************************************
//**************************************
namespace n1x1 {

struct Conv2D1x1OpData {
  Conv2DParams params;
  ExecutionPlan execution_plan;
  nn_conv2d_1x1_plan_t plan;
  nn_conv2d_1x1_job_t *jobs;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

struct Conv2D1x1ThreadData {
  Conv2DThreadData data;
  const nn_conv2d_1x1_plan_t *plan;
  nn_conv2d_1x1_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_1x1_thread_worker(void *context) {
  Conv2D1x1ThreadData *td = (Conv2D1x1ThreadData *)context;
  conv2d_1x1(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
             td->job);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  Conv2D1x1OpData *op = nullptr;
  op = reinterpret_cast<Conv2D1x1OpData *>(
      context->AllocatePersistentBuffer(context, sizeof(Conv2D1x1OpData)));
  op->jobs = nullptr;
  op->stack_scratch_index = -1;
  op->stack_size = 0;
  op->weights_scratch_index = -1;
  op->bias_scratch_index = -1;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op->params,
                       &op->execution_plan);

  // allocate the jobs
  op->jobs =
      reinterpret_cast<nn_conv2d_1x1_job_t *>(context->AllocatePersistentBuffer(
          context, sizeof(nn_conv2d_1x1_job_t) *
                       op->execution_plan.changrps.GetSize() *
                       op->execution_plan.regions.GetSize()));

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2D1x1OpData *op = reinterpret_cast<Conv2D1x1OpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)input->dims->data[1],
                                 (uint32_t)input->dims->data[2],
                                 (uint32_t)input->dims->data[3]};
  nn_image_params_t out_params = {(uint32_t)output->dims->data[1],
                                  (uint32_t)output->dims->data[2],
                                  (uint32_t)output->dims->data[3]};

  int32_t n_jobs = op->execution_plan.changrps.GetSize() *
                   op->execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(op->stack_size, conv2d_shallow_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(weights->data.int8)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetWeightsScratchSize(),
        &op->weights_scratch_index));
  }
  if (IS_NOT_RAM(bso->data.i16)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetBiasScratchSize(),
        &op->bias_scratch_index));
  }

  // set job parameters
  nn_conv2d_1x1_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];
      job_params[i_cg * op->execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {(uint32_t)(region.rows * region.cols), (uint32_t)changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_1x1_init(&op->plan, op->jobs, &in_params, &out_params, &job_params[0],
                  n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2D1x1OpData *op = reinterpret_cast<Conv2D1x1OpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_1x1_thread_worker, stack, op->stack_size);

  // create thread data and tasks
  Conv2D1x1ThreadData thread_data[op->execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the weights and biases
    dispatcher->FetchWeights(&tK, weights->data.int8,
                             op->execution_plan.GetWeightsScratchSize(),
                             changrp);
    dispatcher->FetchBiases(&tBSO, bso->data.i16,
                            op->execution_plan.GetBiasScratchSize(), changrp);

    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * op->execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].data.X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &op->plan;
      op->jobs[i_job].start.K = 0;
      op->jobs[i_job].start.BSO = 0;
      thread_data[i_rg].job = &op->jobs[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace n1x1

//**************************************
//**************************************
//**************************************
// depthwise
//**************************************
//**************************************
//**************************************

namespace depthwise {

struct Conv2DDepthwiseOpData {
  Conv2DParams params;
  ExecutionPlan execution_plan;
  nn_conv2d_depthwise_plan_t plan;
  nn_conv2d_depthwise_job_t *jobs;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

struct Conv2DDepthwiseThreadData {
  Conv2DThreadData data;
  const nn_conv2d_depthwise_plan_t *plan;
  const nn_conv2d_depthwise_job_t *job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_depthwise_thread_worker(void *context) {
  Conv2DDepthwiseThreadData *td = (Conv2DDepthwiseThreadData *)context;
  conv2d_depthwise(td->data.Y, td->data.X, td->data.K, td->data.BSO, td->plan,
                   td->job);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  Conv2DDepthwiseOpData *op = nullptr;
  op = reinterpret_cast<Conv2DDepthwiseOpData *>(
      context->AllocatePersistentBuffer(context,
                                        sizeof(Conv2DDepthwiseOpData)));
  op->jobs = nullptr;
  op->stack_scratch_index = -1;
  op->stack_size = 0;
  op->weights_scratch_index = -1;
  op->bias_scratch_index = -1;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, op->params,
                       &op->execution_plan);

  // allocate the jobs
  op->jobs = reinterpret_cast<nn_conv2d_depthwise_job_t *>(
      context->AllocatePersistentBuffer(
          context, sizeof(nn_conv2d_depthwise_job_t) *
                       op->execution_plan.changrps.GetSize() *
                       op->execution_plan.regions.GetSize()));

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DDepthwiseOpData *op =
      reinterpret_cast<Conv2DDepthwiseOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // set param values not parsed from custom options
  op->params.K_h = weights->dims->data[0];
  op->params.K_w = weights->dims->data[1];

  // setup kernel parameters
  nn_image_params_t in_params = {(uint32_t)input->dims->data[1],
                                 (uint32_t)input->dims->data[2],
                                 (uint32_t)input->dims->data[3]};
  nn_image_params_t out_params = {(uint32_t)output->dims->data[1],
                                  (uint32_t)output->dims->data[2],
                                  (uint32_t)output->dims->data[3]};
  nn_window_params_t conv_window = {
      {(uint32_t)op->params.K_h, (uint32_t)op->params.K_w},
      {-op->params.pad.top, -op->params.pad.left},
      {op->params.stride_h, op->params.stride_w}};

  int32_t n_jobs = op->execution_plan.changrps.GetSize() *
                   op->execution_plan.regions.GetSize();

  // allocate the stack for thread workers
  GET_STACKSIZE(op->stack_size, conv2d_depthwise_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.regions.GetSize(),
      &op->stack_scratch_index));

  // allocate scratch buffers for weights and biases (if necessary)
  if (IS_NOT_RAM(weights->data.int8)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetWeightsScratchSize(),
        &op->weights_scratch_index));
  }
  if (IS_NOT_RAM(bso->data.i16)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetBiasScratchSize(),
        &op->bias_scratch_index));
  }

  // set job parameters
  nn_conv2d_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];
    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      const RowColRegion &region = op->execution_plan.regions[i_rg];
      job_params[i_cg * op->execution_plan.regions.GetSize() + i_rg] = {
          {region.top, region.left, changrp.start},
          {region.rows, region.cols, changrp.size}};
    }
  }

  // initialize the kernel
  conv2d_depthwise_init(&op->plan, op->jobs, &in_params, &out_params,
                        &job_params[0], &conv_window, op->params.pad.zero_point,
                        n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  Conv2DDepthwiseOpData *op =
      reinterpret_cast<Conv2DDepthwiseOpData *>(node->user_data);
  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(conv2d_depthwise_thread_worker, stack,
                              op->stack_size);

  // create thread data and tasks
  Conv2DDepthwiseThreadData thread_data[op->execution_plan.GetNumThreads()];

  // load weights & bias scratch buffers (if necessary)
  int8_t *tK = nullptr;
  int16_t *tBSO = nullptr;

  if (op->weights_scratch_index >= 0) {
    tK = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(tK != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    tBSO = static_cast<int16_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(tBSO != nullptr);
  }

  // fetch the weights
  //   NOTE: They all need to be fetched for each job
  //         This may be changed in the future.
  dispatcher->FetchBuffer(&tK, weights->data.int8,
                          op->execution_plan.GetWeightsScratchSize());

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // fetch the biases
    dispatcher->FetchBiases(&tBSO, bso->data.i16,
                            op->execution_plan.GetBiasScratchSize(), changrp);

    for (int i_rg = 0; i_rg < op->execution_plan.regions.GetSize(); i_rg++) {
      int32_t i_job = i_cg * op->execution_plan.regions.GetSize() + i_rg;
      thread_data[i_rg].data.Y = (nn_image_t *)output->data.int8;
      thread_data[i_rg].data.X = (const nn_image_t *)input->data.int8;
      thread_data[i_rg].data.K = (const nn_tensor_t *)tK;
      thread_data[i_rg].data.BSO = (const nn_bso_block_t *)tBSO;
      thread_data[i_rg].plan = &op->plan;
      op->jobs[i_job].stride.start.BSO = 0;
      thread_data[i_rg].job = &op->jobs[i_job];
      dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_rg]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
  }

  return kTfLiteOk;
}

}  // namespace depthwise
}  // namespace conv

TfLiteRegistration *Register_Conv2D_Deep() {
  static TfLiteRegistration r = {conv::deep::Init, nullptr, conv::deep::Prepare,
                                 conv::deep::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_Shallow() {
  static TfLiteRegistration r = {conv::shallow::Init, nullptr,
                                 conv::shallow::Prepare, conv::shallow::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_1x1() {
  static TfLiteRegistration r = {conv::n1x1::Init, nullptr, conv::n1x1::Prepare,
                                 conv::n1x1::Eval};
  return &r;
}

TfLiteRegistration *Register_Conv2D_Depthwise() {
  static TfLiteRegistration r = {conv::depthwise::Init, nullptr,
                                 conv::depthwise::Prepare,
                                 conv::depthwise::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
