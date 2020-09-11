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
namespace fully_connected {

struct FullyConnectedOpData {
  ExecutionPlan execution_plan;
  nn_fully_connected_plan_t plan;
  nn_fully_connected_job_t* jobs;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

struct FullyConnectedThreadData {
  int16_t* Y;
  const nn_tensor_t* X;
  const nn_tensor_t* W;
  const nn_bso_block_t* BSO;
  nn_fully_connected_plan_t* plan;
  nn_fully_connected_job_t* job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void fully_connected_thread_worker(void* context) {
  FullyConnectedThreadData* td = (FullyConnectedThreadData*)context;
  fully_connected_16(td->Y, td->W, td->X, td->BSO, td->plan, td->job);
}
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  FullyConnectedOpData* op = nullptr;
  context->AllocatePersistentBuffer(context, sizeof(FullyConnectedOpData),
                                    reinterpret_cast<void**>(&op));
  op->jobs = nullptr;
  op->stack_scratch_index = -1;
  op->stack_size = 0;
  op->weights_scratch_index = -1;
  op->bias_scratch_index = -1;

  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, &op->execution_plan);

  // allocate the jobs
  context->AllocatePersistentBuffer(
      context,
      sizeof(nn_fully_connected_job_t) * op->execution_plan.changrps.GetSize(),
      reinterpret_cast<void**>(&op->jobs));

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* weights = GetInput(context, node, 1);
  const TfLiteTensor* bso = GetInput(context, node, 2);

  int32_t C_in = weights->dims->data[1];
  int32_t C_out = weights->dims->data[0];

  FullyConnectedOpData* op =
      reinterpret_cast<FullyConnectedOpData*>(node->user_data);

  // allocate the stack for thread workers
  GET_STACKSIZE(op->stack_size, fully_connected_thread_worker);
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
  size_t n_jobs = op->execution_plan.changrps.GetSize();
  nn_fully_connected_job_params_t job_params[n_jobs];

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup& changrp = op->execution_plan.changrps[i_cg];
    job_params[i_cg] = {(uint32_t)changrp.start, (channel_count_t)changrp.size};
  }

  // initialize the kernel
  fully_connected_init(&op->plan, op->jobs, C_in, C_out, &job_params[0],
                       n_jobs);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* weights = GetInput(context, node, 1);
  const TfLiteTensor* bso = GetInput(context, node, 2);
  TfLiteTensor* output = GetOutput(context, node, 0);

  FullyConnectedOpData* op =
      reinterpret_cast<FullyConnectedOpData*>(node->user_data);
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(fully_connected_thread_worker, stack,
                              op->stack_size);

  // create thread data and tasks
  int i_th = 0;
  int n_th = op->execution_plan.GetNumThreads();
  FullyConnectedThreadData thread_data[n_th];

  // load weights & bias scratch buffers (if necessary)
  size_t weights_load_offset = 0;
  size_t biases_load_offset = 0;
  size_t weights_fetch_size;
  // int8_t* tW[n_th];
  // int16_t* tBSO[n_th];
  // std::memset(tW, 0, n_th * sizeof(int8_t*));
  // std::memset(tBSO, 0, n_th * sizeof(int16_t*));
  int8_t *sW, *tW;
  int16_t *sBSO, *tBSO;

  if (op->weights_scratch_index >= 0) {
    sW = static_cast<int8_t*>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(sW != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    sBSO = static_cast<int16_t*>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(sBSO != nullptr);
  }

  weights_fetch_size = std::min(
      (size_t)(changrp_len * op->execution_plan.GetWeightsScratchSize() /
               (op->execution_plan.changrps[n_th - 1].start +
                op->execution_plan.changrps[n_th - 1].size)),
      op->execution_plan.GetWeightsScratchSize());

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.GetSize(); i_cg++) {
    const ChannelGroup& changrp = op->execution_plan.changrps[i_cg];

    // offset into the temp W and BSO pointers based on how many bytes we have
    // loaded since the last JoinTasks
    tW = sW + weights_load_offset;
    tBSO = sBSO + biases_load_offset;

    // fetch the weights and biases
    weights_load_offset += dispatcher->FetchWeights(
        &tW, weights->data.int8, weights_fetch_size, changrp);

    biases_load_offset += dispatcher->FetchBiases(
        &tBSO, bso->data.i16, op->execution_plan.GetBiasScratchSize(), changrp);

    thread_data[i_th].Y = output->data.i16;
    thread_data[i_th].X = input->data.int8;
    thread_data[i_th].W = tW;
    thread_data[i_th].BSO = (const nn_bso_block_t*)tBSO;
    thread_data[i_th].plan = &op->plan;
    op->jobs[i_cg].stride.start.W = 0;
    op->jobs[i_cg].stride.start.BSO = 0;
    thread_data[i_th].job = &op->jobs[i_cg];
    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_th]));

    i_th++;

    if (i_th == n_th) {
      dispatcher->JoinTasks();
      i_th = 0;
      weights_load_offset = 0;
      biases_load_offset = 0;
    }
  }
  dispatcher->JoinTasks();  // finish up any added tasks

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
