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
  // nn_fully_connected_plan_t plan;
  // nn_fully_connected_job_t* jobs;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

struct FullyConnectedThreadData {
  int8_t* Y;
  const nn_tensor_t* X;
  const nn_tensor_t* W;
  const nn_bso_block_t* BSO;
  channel_count_t C_in;
  channel_count_t C_out_start;
  channel_count_t C_out_end;
  // nn_fully_connected_plan_t* plan;
  // nn_fully_connected_job_t* job;
};

// void* Init_8(TfLiteContext* context, const char* buffer, size_t length) {
//   FullyConnectedOpData* op = nullptr;
//   op = reinterpret_cast<FullyConnectedOpData*>(
//       context->AllocatePersistentBuffer(context,
//       sizeof(FullyConnectedOpData)));

//   TFLITE_DCHECK(buffer != nullptr);
//   parse_custom_options(context, buffer, length, &op->execution_plan);

//   return op;
// }

// TfLiteStatus Prepare_8(TfLiteContext* context, TfLiteNode* node) {
//   TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
//   TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

//   return kTfLiteOk;
// }

// TfLiteStatus Eval_8(TfLiteContext* context, TfLiteNode* node) {
//   const TfLiteTensor* input = GetInput(context, node, 0);
//   const TfLiteTensor* weights = GetInput(context, node, 1);
//   const TfLiteTensor* bso = GetInput(context, node, 2);
//   TfLiteTensor* output = GetOutput(context, node, 0);

//   FullyConnectedOpData* op =
//       reinterpret_cast<FullyConnectedOpData*>(node->user_data);

//   int32_t C_in = weights->dims->data[1];
//   int32_t C_out = weights->dims->data[0];

//   fully_connected_8(output->data.int8, weights->data.int8, input->data.int8,
//                     (const nn_bso_block_t*)bso->data.i16, C_in, 0, C_out);
//   return kTfLiteOk;
// }

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void fully_connected_thread_worker(void* context) {
  FullyConnectedThreadData* td = (FullyConnectedThreadData*)context;
  fully_connected_8(td->Y, td->W, td->X, td->BSO, td->C_in, td->C_out_start,
                    td->C_out_end);
}
}

void* Init_8(TfLiteContext* context, const char* buffer, size_t length) {
  FullyConnectedOpData* op = nullptr;
  op = reinterpret_cast<FullyConnectedOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(FullyConnectedOpData)));
  op->stack_scratch_index = -1;
  op->stack_size = 0;
  op->weights_scratch_index = -1;
  op->bias_scratch_index = -1;

  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, &op->execution_plan);

  return op;
}

TfLiteStatus Prepare_8(TfLiteContext* context, TfLiteNode* node) {
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

  return kTfLiteOk;
}

TfLiteStatus Eval_8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* weights = GetInput(context, node, 1);
  const TfLiteTensor* bso = GetInput(context, node, 2);
  TfLiteTensor* output = GetOutput(context, node, 0);

  FullyConnectedOpData* op =
      reinterpret_cast<FullyConnectedOpData*>(node->user_data);

  int32_t C_in = weights->dims->data[1];

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

  // load weights & bias scratch buffers(if necessary)
  size_t weights_load_offset = 0;
  size_t biases_load_offset = 0;
  size_t weights_fetch_size;
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

    // offset into the temp W and BSO pointers based on how many bytes we
    // have loaded since the last JoinTasks
    tW = sW + weights_load_offset;
    tBSO = sBSO + biases_load_offset;

    // fetch the weights and biases
    weights_load_offset += dispatcher->FetchWeights(
        &tW, weights->data.int8, weights_fetch_size, changrp);

    biases_load_offset += dispatcher->FetchBiases(
        &tBSO, bso->data.i16, op->execution_plan.GetBiasScratchSize(), changrp);

    thread_data[i_th].Y = &output->data.int8[changrp.start];
    thread_data[i_th].X = input->data.int8;
    thread_data[i_th].W = tW;
    thread_data[i_th].BSO = (const nn_bso_block_t*)tBSO;
    thread_data[i_th].C_in = C_in;
    thread_data[i_th].C_out_start = 0;
    thread_data[i_th].C_out_end = thread_data[i_th].C_out_start + changrp.size;
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

TfLiteRegistration* Register_FullyConnected_8() {
  static TfLiteRegistration r = {fully_connected::Init_8, nullptr,
                                 fully_connected::Prepare_8,
                                 fully_connected::Eval_8};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
