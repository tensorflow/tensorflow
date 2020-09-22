

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "flatbuffers/flexbuffers.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

//**************************************
//**************************************
//**************************************
// BSign
//**************************************
//**************************************
//**************************************
namespace bsign {

struct BSign8OpData {
    nn_bsign_8_plan_t plan;
    nn_bsign_8_job_t* jobs;
    unsigned n_threads;
    size_t stack_size; //The amount of stack required to run an instance of bsign_8_thread_worker
    int stack_scratch_index; //The index where the above stack will be allocated
};

struct BSign8ThreadData {
  int32_t* Y;
  const int8_t* X;
  nn_bsign_8_plan_t* plan;
  nn_bsign_8_job_t* job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void bsign_8_thread_worker(void* context) {

  TFLITE_DCHECK(context != nullptr);
  BSign8ThreadData* td = (BSign8ThreadData*)context;
  TFLITE_DCHECK(td->Y != nullptr);
  TFLITE_DCHECK(td->X != nullptr);
  bsign_8(td->Y, td->X, td->plan, td->job);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length)
{
    BSign8OpData *op = reinterpret_cast<BSign8OpData *>(context->AllocatePersistentBuffer(context, sizeof(BSign8OpData)));

    op->jobs = nullptr; 
    op->stack_scratch_index = -1;
    op->stack_size = 0;

    //TODO get n_threads from custom options  
    //TFLITE_DCHECK(buffer != nullptr);
    /* Retrieve custom options */
    //op->n_threads = named_uint32_custom_option(context, buffer, length, "th");
    op->n_threads = 1;

    /* Allocate the jobs */
    op->jobs = reinterpret_cast<nn_bsign_8_job_t*>(context->AllocatePersistentBuffer(context, sizeof(nn_bsign_8_job_t*) * op->n_threads));

    return op;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {

    TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    const TfLiteTensor* input = GetInput(context, node, 0);
    const int32_t inputLength = input->bytes / sizeof(int8_t);

    BSign8OpData* op = reinterpret_cast<BSign8OpData*>(node->user_data);

    /* Allocate the stack for thread workers */
    GET_STACKSIZE(op->stack_size, bsign_8_thread_worker);
    context->RequestScratchBufferInArena(context, op->stack_size * op->n_threads, &op->stack_scratch_index);
    TFLITE_DCHECK(op->stack_scratch_index != -1);

    /* Prepare the kernel */
    bsign_8_prepare(&op->plan, op->jobs, inputLength, input->params.zero_point, op->n_threads);

    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  
    const TfLiteTensor* input = GetInput(context, node, 0);
    TfLiteTensor* output = GetOutput(context, node, 0);
    const int32_t length = input->bytes / sizeof(int8_t);

    BSign8OpData* op = reinterpret_cast<BSign8OpData*>(node->user_data);
    
    Dispatcher* dispatcher = GetDispatcher();

    // initialize the dispatcher
    char* stack = static_cast<char*>(context->GetScratchBuffer(context, op->stack_scratch_index));
   
    TFLITE_DCHECK(stack != nullptr);
    dispatcher->InitializeTasks(bsign_8_thread_worker, stack, op->stack_size);

    // create thread data and tasks
    BSign8ThreadData thread_data[op->n_threads];

    for (int i = 0; i < op->n_threads; i++)
    {
        thread_data[i].Y = nullptr;
        thread_data[i].X = nullptr;
        thread_data[i].job = nullptr;
        thread_data[i].plan = nullptr;
    }

    for (int i_job = 0; i_job < op->n_threads; i_job++) {
        thread_data[i_job].Y = (int32_t*)output->data.i32;
        thread_data[i_job].X = (int8_t*)input->data.int8;
        thread_data[i_job].job = &op->jobs[i_job];
        thread_data[i_job].plan = &op->plan;
        dispatcher->AddTask(reinterpret_cast<void*>(&thread_data[i_job]));
    }
    // start and wait for tasks to complete
    dispatcher->JoinTasks();
    
    return kTfLiteOk;
}




}  // namespace bsign

TfLiteRegistration *Register_BSign_8() {
   static TfLiteRegistration r = {bsign::Init, nullptr,
                                  bsign::Prepare, bsign::Eval};
   return &r;
 }



}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
