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
namespace conv {

//**************************************
//**************************************
//**************************************
// BConv2D binary output
//**************************************
//**************************************
//**************************************
namespace binary_out {

//These are the bconv2d properties common to all threads
struct BConv2DThreadDataCommon {
  nn_image_t * Y;
  nn_image_t * X;
  nn_tensor_t * K;
  int32_t * thresholds;

  nn_image_params_t x;
  nn_image_params_t y;
  nn_window_params_t k; 
};

//These are the bconv2d properties unique to each thread
struct BConv2DThreadData {
  BConv2DThreadDataCommon * common;
  //This describes the region that that thread will process
  uint32_t y_loc_x;
  uint32_t y_loc_y;
  uint32_t y_sub_width;
  uint32_t y_sub_height;

  BConv2DThreadData * next;

};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void bconv2d_thread_generator(void *context) {
  BConv2DThreadData *td = (BConv2DThreadData *)context;

  while(td != nullptr){

    bnn_conv2d_bin_out_valid((bnn_b32_t*) td->common->Y,
      (const bnn_b256_t*) td->common->X, 
      (const bnn_b256_t*) td->common->K, 
      (const int32_t*) td->common->thresholds,
      (const nn_image_params_t*) &td->common->x, 
      (const nn_image_params_t*) &td->common->y,
      (const nn_window_params_t*)&td->common->k, 
      (const unsigned) td->y_loc_x, 
      (const unsigned) td->y_loc_y,
      (const unsigned) td->y_sub_width, 
      (const unsigned) td->y_sub_height);

    td = td->next;
  } 

}
}

/*
This is a struct that describes the memory required to configure the operator.
*/
struct BConv2DBinOutOpData {

  //Data that is common to all threads processing the bconv2d
  BConv2DThreadDataCommon common;

  //These are the head pointers to the regions a thread will have to do.
  //i.e. jobs[0] points to the first region that thread 0 will have to process,
  //jobs[1] points to the first region thread 1 will have to process, etc.
  BConv2DThreadData ** jobs; 

  //The actual memory used to describe the regions threads will have to process. 
  BConv2DThreadData * regions; 

  //The number of concurrent instances of bconv2d_thread_generator
  unsigned n_threads;

  //The total number of regions(jobs) processed by the threads, i.e. 6 regions could be
  //processed to 5 threads with 4 threads doing 1 region and one doing 2 regions. 
  unsigned n_regions;
  
  size_t stack_size; //The amount of stack required to run an instance of bconv2d_thread_generator
  int stack_scratch_index; //The index where the above stack will be allocated

};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  BConv2DBinOutOpData *op = reinterpret_cast<BConv2DBinOutOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(BConv2DBinOutOpData)));

  op->stack_scratch_index = -1;
  op->stack_size = 0;

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  TFLITE_DCHECK(length > 0); //in fact it must be at least 6x uint32_t big

  op->common.k.stride.horizontal = get_named_uint32_custom_option(context, buffer, length, "stride_width");
  op->common.k.stride.vertical = get_named_uint32_custom_option(context, buffer, length, "stride_height");
  op->common.k.dilation.horizontal = get_named_uint32_custom_option(context, buffer, length, "dilation_width_factor");
  op->common.k.dilation.vertical = get_named_uint32_custom_option(context, buffer, length, "dilation_height_factor");
  //op->n_threads = get_named_uint32_custom_option(context, buffer, length, "n_threads");
  //op->n_regions = get_named_uint32_custom_option(context, buffer, length, "n_regions");

  // TODO - requires parallelisation pass in xformer
  op->n_threads = 1;
  op->n_regions = 1;


  TFLITE_DCHECK(op->n_threads > 0);
  TFLITE_DCHECK(op->n_regions > 0);
  TFLITE_DCHECK(op->common.k.stride.horizontal > 0);
  TFLITE_DCHECK(op->common.k.stride.vertical > 0);
  TFLITE_DCHECK(op->common.k.dilation.horizontal > 0);
  TFLITE_DCHECK(op->common.k.dilation.vertical > 0);

  // Allocate the jobs (one pointer per thread)
  op->jobs = reinterpret_cast<BConv2DThreadData **>(
    context->AllocatePersistentBuffer(context, 
    sizeof(BConv2DThreadData*) * op->n_threads));

  // Allocate the regions (one BConv2DThreadData per region)                                  
  op->regions = reinterpret_cast<BConv2DThreadData *>(
    context->AllocatePersistentBuffer(context,
    sizeof(BConv2DThreadData) * op->n_regions));

  //Init the job head pointers to a null pointer 
  for (unsigned i=0;i<op->n_threads;++i){
    op->jobs[i] = nullptr;
  }

  //Fill the regions and the job heads
  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();
  auto keys = map.Keys();
  auto values = map.Values();

  unsigned region_idx=0;
  for (int i = 0; i < map.size(); ++i) {
    const std::string &key = keys[i].AsString().str();
    if (key.compare("region") == 0) {
      //values represent [thread_idx, y_loc_x, y_loc_y, y_sub_width, y_sub_height]
      const auto &vec =
            values[i].AsVector();  

        BConv2DThreadData * r = &(op->regions[region_idx]);
        r->common = &op->common;
        unsigned thread_idx = vec[0].AsInt32();
        r->y_loc_x = vec[1].AsInt32();
        r->y_loc_y = vec[2].AsInt32();
        r->y_sub_width = vec[3].AsInt32();
        r->y_sub_height = vec[4].AsInt32();
        r->next = op->jobs[thread_idx];
        op->jobs[thread_idx] = r;
        region_idx++;
    }
  }

  TFLITE_DCHECK((op->n_regions-1) == region_idx);

  return op;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *kernel = GetInput(context, node, 1);
  const TfLiteTensor *thresholds = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  BConv2DBinOutOpData *op =
      reinterpret_cast<BConv2DBinOutOpData *>(node->user_data);

  // setup runtime parameters
  op->common.X = (nn_image_t *) input->data.int8;
  op->common.x.height = (uint32_t)input->dims->data[1];
  op->common.x.width = (uint32_t)input->dims->data[2];
  op->common.x.channels = (uint32_t)input->dims->data[3];

  op->common.Y = (nn_image_t *) output->data.int8;
  op->common.y.height = (uint32_t)output->dims->data[1];
  op->common.y.width = (uint32_t)output->dims->data[2];
  op->common.y.channels = (uint32_t)output->dims->data[0];

  // FIXME *32
  TF_LITE_ENSURE_EQ(context, (uint32_t)kernel->dims->data[0], (uint32_t)output->dims->data[3]*32);

  op->common.K = (nn_tensor_t *) kernel->data.int8;
  op->common.k.shape.height = (uint32_t)kernel->dims->data[1];
  op->common.k.shape.width = (uint32_t)kernel->dims->data[2];

  op->common.thresholds = (int32_t*)thresholds->data.int8;

  // allocate the stack for thread workers
  GET_STACKSIZE(op->stack_size, bconv2d_thread_generator);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->n_threads,
      &op->stack_scratch_index));

  TF_LITE_ENSURE(context, op->stack_scratch_index != -1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  BConv2DBinOutOpData *op =
      reinterpret_cast<BConv2DBinOutOpData *>(node->user_data);

  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));

  TF_LITE_ENSURE(context, stack != nullptr);
  dispatcher->InitializeTasks(bconv2d_thread_generator, stack,
                              op->stack_size);

  for (int region_idx = 0; region_idx < op->n_threads; region_idx++)
    dispatcher->AddTask(reinterpret_cast<void *>(op->jobs[region_idx]));
  // start and wait for tasks to complete
  dispatcher->JoinTasks();

  return kTfLiteOk;
}

}  // namespace binary_out

}  // namespace conv


TfLiteRegistration *Register_BConv2D_Bin_Out() {
  static TfLiteRegistration r = {conv::binary_out::Init, nullptr,
                                 conv::binary_out::Prepare, conv::binary_out::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
