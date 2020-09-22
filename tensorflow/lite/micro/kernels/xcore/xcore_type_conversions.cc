#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_planning.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace type_conversions {

struct RequantizeOpData {
  ExecutionPlan execution_plan;
  nn_requantize_16_to_8_job_t* jobs;
  int stack_scratch_index;
  size_t stack_size;
};

struct RequantizeThreadData {
  int8_t* Y;
  const int16_t* X;
  nn_requantize_16_to_8_job_t* job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void requantize_16_to_8_thread_worker(void* context) {
  RequantizeThreadData* td = (RequantizeThreadData*)context;
  requantize_16_to_8(td->Y, td->X, td->job);
}
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  RequantizeOpData* op = nullptr;
  op = reinterpret_cast<RequantizeOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(RequantizeOpData)));
  op->jobs = nullptr;
  op->stack_scratch_index = -1;
  op->stack_size = 0;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, &op->execution_plan);

  // allocate the jobs
  op->jobs = reinterpret_cast<nn_requantize_16_to_8_job_t*>(
      context->AllocatePersistentBuffer(
          context, sizeof(nn_requantize_16_to_8_job_t) *
                       op->execution_plan.GetNumThreads()));

  return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  int32_t length = input->bytes / sizeof(int16_t);

  RequantizeOpData* op = reinterpret_cast<RequantizeOpData*>(node->user_data);

  // allocate the stack for thread workers
  GET_STACKSIZE(op->stack_size, requantize_16_to_8_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  // initialize the kernel
  requantize_16_to_8_init(op->jobs, length, op->execution_plan.GetNumThreads());

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  int32_t length = input->bytes / sizeof(int16_t);

  RequantizeOpData* op = reinterpret_cast<RequantizeOpData*>(node->user_data);
  Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  char* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(requantize_16_to_8_thread_worker, stack,
                              op->stack_size);

  // create thread data and tasks
  RequantizeThreadData thread_data[op->execution_plan.GetNumThreads()];

  for (int i_job = 0; i_job < op->execution_plan.GetNumThreads(); i_job++) {
    thread_data[i_job].Y = output->data.int8;
    thread_data[i_job].X = input->data.i16;
    thread_data[i_job].job = &op->jobs[i_job];
    dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_job]));
  }
  // start and wait for tasks to complete
  dispatcher->JoinTasks();

  return kTfLiteOk;
}

}  // namespace type_conversions

TfLiteRegistration* Register_Requantize_16_to_8() {
  static TfLiteRegistration r = {type_conversions::Init, nullptr,
                                 type_conversions::Prepare,
                                 type_conversions::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
