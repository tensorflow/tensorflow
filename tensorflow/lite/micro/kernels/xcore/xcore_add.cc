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
namespace add {

struct AddOpData {
  ExecutionPlan execution_plan;
  int stack_scratch_index;
  size_t stack_size;
};

struct AddThreadData {
  int8_t* Y;
  const int8_t* X0;
  const int8_t* X1;
  // const nn_add_params_t* params;
  int32_t start;
  int32_t count;
};

// extern "C" {
// ATTRIBUTE_THREAD_FUNCTION void add_thread_worker(void* context) {
//   AddThreadData* td = (AddThreadData*)context;
//   add_elementwise(td->Y, td->X0, td->X0, td->params, td->start, td->count);
// }
// }

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  AddOpData* op = nullptr;
  op = reinterpret_cast<AddOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(AddOpData)));
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

  AddOpData* op = reinterpret_cast<AddOpData*>(node->user_data);

  // allocate the stack for thread workers
  // GET_THREAD_FUNCTION_STACKSIZE(op->stack_size, add_thread_worker);
  // TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
  //     context, op->stack_size * op->execution_plan.GetNumThreads(),
  //     &op->stack_scratch_index));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input0 = GetInput(context, node, 0);
  const TfLiteTensor* input1 = GetInput(context, node, 1);
  TfLiteTensor* output = GetOutput(context, node, 0);

  AddOpData* op = reinterpret_cast<AddOpData*>(node->user_data);
  // Dispatcher* dispatcher = GetDispatcher();

  // initialize the dispatcher
  // char* stack = static_cast<char*>(
  //     context->GetScratchBuffer(context, op->stack_scratch_index));
  // TFLITE_DCHECK(stack != nullptr);
  // dispatcher->InitializeTasks(add_thread_worker, stack, op->stack_size);

  // create thread data
  // int n_th = op->execution_plan.GetNumThreads();
  // AddThreadData thread_data[n_th];

  // TODO: setup params, see lib_nn/api/nn_layers.h

  // create tasks
  // for (int i_sl = 0; i_sl < op->execution_plan.slices.GetSize(); i_sl++)
  // {
  //   const Slice& slice = op->execution_plan.slices[i_sl];
  //   thread_data[i_sl].data.Y = output->data.int8;
  //   thread_data[i_sl].data.X0 = input0->data.int8;
  //   thread_data[i_sl].data.X1 = input1->data.int8;
  //   thread_data[i_rg].params = &TBD;
  //   thread_data[i_rg].start = slice.start;
  //   thread_data[i_rg].count = slice.size;

  //   dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_sl]));
  // }

  // start and wait for tasks to complete
  // dispatcher->JoinTasks();

  memset(output->data.raw, 0, output->bytes);

  return kTfLiteOk;
}

}  // namespace add

TfLiteRegistration* Register_Add_8() {
  static TfLiteRegistration r = {add::Init, nullptr, add::Prepare, add::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
