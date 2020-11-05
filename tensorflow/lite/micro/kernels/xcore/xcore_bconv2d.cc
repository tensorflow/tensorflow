
#include "flatbuffers/flexbuffers.h"
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

//**************************************
//**************************************
//**************************************
// BConv2D binary output
//**************************************
//**************************************
//**************************************
namespace bitpacked {

// These are the bconv2d properties common to all threads
struct BConv2DThreadDataCommon {
  bnn_b32_t *Y;
  const bnn_b32_t *X;
  const bnn_b32_t *K;
  const int32_t *thresholds;

  nn_image_params_t x;
  nn_image_params_t y;
  nn_window_params_t k;
};

// These are the bconv2d properties unique to each thread
struct BConv2DThreadData {
  BConv2DThreadDataCommon *common;
  // This describes the region that that thread will process
  uint32_t y_loc_x;
  uint32_t y_loc_y;
  uint32_t y_sub_width;
  uint32_t y_sub_height;

  BConv2DThreadData *next;
};

namespace deepin {

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void bconv2d_bitpacked_deepin_thread_generator(
    void *context) {
  auto *td = static_cast<BConv2DThreadData *>(context);

  while (td != nullptr) {
    bnn_conv2d_bin_out_valid(
        td->common->Y, (bnn_b256_t *)td->common->X, (bnn_b256_t *)td->common->K,
        td->common->thresholds, &td->common->x, &td->common->y, &td->common->k,
        td->y_loc_x, td->y_loc_y, td->y_sub_width, td->y_sub_height);

    td = td->next;
  }
}
}

/*
This is a struct that describes the memory required to configure the operator.
*/
struct BConv2DBitpackedOpData {
  // Data that is common to all threads processing the bconv2d
  BConv2DThreadDataCommon common;

  // These are the head pointers to the regions a thread will have to do.
  // i.e. jobs[0] points to the first region that thread 0 will have to process,
  // jobs[1] points to the first region thread 1 will have to process, etc.
  BConv2DThreadData **jobs;

  // The actual memory used to describe the regions threads will have to
  // process.
  BConv2DThreadData *regions;

  // The number of concurrent instances of
  // bconv2d_bitpacked_deepin_thread_generator
  unsigned n_threads;

  // The total number of regions(jobs) processed by the threads, i.e. 6 regions
  // could be processed to 5 threads with 4 threads doing 1 region and one doing
  // 2 regions.
  unsigned n_regions;

  ExecutionPlan execution_plan;

  size_t stack_size;  // The amount of stack required to run an instance of
                      // bconv2d_bitpacked_deepin_thread_generator
  int stack_scratch_index;  // The index where the above stack will be allocated

  // for loading from external mem
  int weights_scratch_index = 0;
  int bias_scratch_index = 0;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  BConv2DBitpackedOpData *op_data = reinterpret_cast<BConv2DBitpackedOpData *>(
      context->AllocatePersistentBuffer(context,
                                        sizeof(BConv2DBitpackedOpData)));

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  TFLITE_DCHECK(length > 0);  // in fact it must be at least 6x uint32_t big

  flexbuffers::Vector Kshape =
      get_named_uint32_custom_option_vector(context, buffer, length, "K");
  op_data->common.y.channels = Kshape[0].AsUInt32();
  op_data->common.k.shape.height = Kshape[1].AsUInt32();
  op_data->common.k.shape.width = Kshape[2].AsUInt32();
  op_data->common.x.channels = Kshape[3].AsUInt32();

  flexbuffers::Vector strides =
      get_named_uint32_custom_option_vector(context, buffer, length, "stride");
  op_data->common.k.stride.vertical = strides[0].AsInt32();
  op_data->common.k.stride.horizontal = strides[1].AsInt32();

  op_data->common.k.dilation.horizontal = 1;
  op_data->common.k.dilation.vertical = 1;

  // TODO
  // parse_custom_options(context, buffer, length, &op_data->execution_plan);
  op_data->n_threads = 1;
  op_data->n_regions = 1;

  TFLITE_DCHECK(op_data->n_threads > 0);
  TFLITE_DCHECK(op_data->n_regions > 0);
  TFLITE_DCHECK(op_data->common.k.stride.horizontal > 0);
  TFLITE_DCHECK(op_data->common.k.stride.vertical > 0);
  TFLITE_DCHECK(op_data->common.k.dilation.horizontal > 0);
  TFLITE_DCHECK(op_data->common.k.dilation.vertical > 0);

  // Allocate the jobs (one pointer per thread)
  op_data->jobs =
      reinterpret_cast<BConv2DThreadData **>(context->AllocatePersistentBuffer(
          context, sizeof(BConv2DThreadData *) * op_data->n_threads));

  // Allocate the regions (one BConv2DThreadData per region)
  op_data->regions =
      reinterpret_cast<BConv2DThreadData *>(context->AllocatePersistentBuffer(
          context, sizeof(BConv2DThreadData) * op_data->n_regions));

  auto region_idx = 0;
  auto *r = &(op_data->regions[region_idx]);
  r->common = &op_data->common;
  r->y_loc_x = 0;
  r->y_loc_y = 0;
  r->next = nullptr;
  op_data->jobs[region_idx] = r;

  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto *op_data = reinterpret_cast<BConv2DBitpackedOpData *>(node->user_data);

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *kernel = GetInput(context, node, 1);
  const TfLiteTensor *thresholds = GetInput(context, node, 2);
  const TfLiteTensor *output = GetOutput(context, node, 0);

  // setup runtime parameters
  op_data->common.x.height = (uint32_t)input->dims->data[1];
  op_data->common.x.width = (uint32_t)input->dims->data[2];
  // FIXME *32
  TF_LITE_ENSURE_EQ(context, op_data->common.x.channels,
                    (uint32_t)input->dims->data[3] * 32);

  op_data->common.y.height = (uint32_t)output->dims->data[1];
  op_data->common.y.width = (uint32_t)output->dims->data[2];
  // FIXME *32
  TF_LITE_ENSURE_EQ(context, op_data->common.y.channels,
                    (uint32_t)output->dims->data[3] * 32);

  // TODO: remove this when parallelization is done
  op_data->regions[0].y_sub_width = op_data->common.y.width;
  op_data->regions[0].y_sub_height = op_data->common.y.height;

  // TODO: fix this this when parallelization is done
  // allocate scratch buffers for weights and biases (if necessary)
  if (!is_ram_address((uintptr_t)kernel->data.data)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, kernel->bytes, &op_data->weights_scratch_index));
  }
  if (!is_ram_address((uintptr_t)thresholds->data.data)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, thresholds->bytes, &op_data->bias_scratch_index));
  }

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op_data->stack_size,
                                bconv2d_bitpacked_deepin_thread_generator);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->n_threads,
      &op_data->stack_scratch_index));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = reinterpret_cast<BConv2DBitpackedOpData *>(node->user_data);

  Dispatcher *dispatcher = GetDispatcher();

  const TfLiteTensor *input = GetInput(context, node, 0);
  const TfLiteTensor *kernel = GetInput(context, node, 1);
  const TfLiteTensor *thresholds = GetInput(context, node, 2);
  TfLiteTensor *output = GetOutput(context, node, 0);

  // setup runtime pointers
  op_data->common.X = GetTensorData<bnn_b32_t>(input);
  op_data->common.Y = GetTensorData<bnn_b32_t>(output);

  // load weights & bias scratch buffers (if necessary)
  if (op_data->weights_scratch_index >= 0) {
    op_data->common.K = static_cast<const bnn_b32_t *>(
        context->GetScratchBuffer(context, op_data->weights_scratch_index));
    dispatcher->FetchBuffer((int8_t **)&op_data->common.K,
                            GetTensorData<int8_t>(kernel), kernel->bytes);
  } else {
    op_data->common.K = GetTensorData<bnn_b32_t>(kernel);
  }
  TF_LITE_ENSURE(context, op_data->common.K);

  if (op_data->bias_scratch_index >= 0) {
    op_data->common.thresholds = static_cast<const int32_t *>(
        context->GetScratchBuffer(context, op_data->bias_scratch_index));
    dispatcher->FetchBuffer((int8_t **)&op_data->common.thresholds,
                            GetTensorData<int8_t>(thresholds),
                            thresholds->bytes);
  } else {
    op_data->common.thresholds = GetTensorData<int32_t>(thresholds);
  }
  TF_LITE_ENSURE(context, op_data->common.thresholds);

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));

  TF_LITE_ENSURE(context, stack != nullptr);
  dispatcher->InitializeTasks(bconv2d_bitpacked_deepin_thread_generator, stack,
                              op_data->stack_size);

  for (int region_idx = 0; region_idx < op_data->n_threads; region_idx++)
    dispatcher->AddTask(reinterpret_cast<void *>(op_data->jobs[region_idx]));
  // start and wait for tasks to complete
  dispatcher->JoinTasks();

  return kTfLiteOk;
}

}  // namespace deepin

}  // namespace bitpacked

}  // namespace conv

TfLiteRegistration *Register_BConv2D_Bitpacked_Deepin() {
  static TfLiteRegistration r = {conv::bitpacked::deepin::Init, nullptr,
                                 conv::bitpacked::deepin::Prepare,
                                 conv::bitpacked::deepin::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
