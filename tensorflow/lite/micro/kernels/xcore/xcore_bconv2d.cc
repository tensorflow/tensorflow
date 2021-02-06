
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
namespace bconv {

struct BConv2DJob : RowColRegion {
  BConv2DJob *next = nullptr;
};

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct BConv2DArguments {
  const bnn_b32_t *X;
  const bnn_b32_t *K;

  nn_image_params_t x;
  nn_image_params_t y;
  nn_window_params_t k;

  union {
    bnn_b32_t *Y_bitpacked;
    int8_t *Y_int8;
  };

  union {
    const int32_t *thresholds;     // used in bitpacked only
    const int16_t *accu_modifier;  // used in generic int8 only
  };

  // for int8 only
  const int16_t *post_act_mult;
  const int16_t *post_act_bias;
  const output_transform_values_t *output_trf_parameters;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct BConv2DThreadData {
  BConv2DJob *job;  // This describes the regions that that thread will process
  int thread_scratch_idx;
  bnn_b32_t *thread_scratch;  // size should be K_h * K_w * C_in / 32 + 8
  BConv2DArguments *args;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void bconv2d_bitpacked_deepin_thread_worker(
    void *context) {
  auto *td = static_cast<BConv2DThreadData *>(context);
  auto *job = td->job;
  while (job) {
    bconv2d_bin_DI_valid(td->args->Y_bitpacked, (const bnn_b256_t *)td->args->X,
                         (const bnn_b256_t *)td->args->K, td->args->thresholds,
                         &td->args->x, &td->args->y, &td->args->k, job->left,
                         job->top, job->cols, job->rows);
    job = job->next;
  }
}

ATTRIBUTE_THREAD_FUNCTION void bconv2d_bitpacked_thread_worker(void *context) {
  auto *td = static_cast<BConv2DThreadData *>(context);
  auto *job = td->job;
  while (job) {
    bconv2d_bin_valid(td->args->Y_bitpacked, td->args->X, td->args->K,
                      td->args->thresholds, td->thread_scratch, &td->args->x,
                      &td->args->y, &td->args->k, job->left, job->top,
                      job->cols, job->rows);
    job = job->next;
  }
}

ATTRIBUTE_THREAD_FUNCTION void bconv2d_int8_deepin_deepout_thread_worker(
    void *context) {
  auto *td = static_cast<BConv2DThreadData *>(context);
  auto *job = td->job;
  while (job) {
    bconv2d_int8_DIDO_valid(
        td->args->Y_int8, (const bnn_b256_t *)td->args->X,
        (const bnn_b256_t *)td->args->K, td->args->post_act_mult,
        td->args->post_act_bias, td->args->output_trf_parameters, &td->args->x,
        &td->args->y, &td->args->k, job->left, job->top, job->cols, job->rows);
    job = job->next;
  }
}

ATTRIBUTE_THREAD_FUNCTION void bconv2d_int8_thread_worker(void *context) {
  auto *td = static_cast<BConv2DThreadData *>(context);
  auto *job = td->job;
  while (job) {
    bconv2d_int8_valid(td->args->Y_int8, td->args->X, td->args->K,
                       td->args->post_act_mult, td->args->post_act_bias,
                       td->args->accu_modifier, td->args->output_trf_parameters,
                       td->thread_scratch, &td->args->x, &td->args->y,
                       &td->args->k, job->left, job->top, job->cols, job->rows);
    job = job->next;
  }
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

struct BConv2DOpData {
  // Data that is args to all threads processing the bconv2d
  BConv2DArguments args;

  // These are the pointers to the thread data the threads will have to use.
  BConv2DThreadData *threads;

  // The actual memory used to describe the jobs (regions) threads will have
  // to
  // process.
  BConv2DJob *jobs;

  // The number of concurrent instances of
  // bconv2d_*_thread_generator
  unsigned n_threads;

  // The total number of jobs (regions) processed by the threads, i.e. 6
  // regions could be processed to 5 threads with 4 threads doing 1 region and
  // one doing 2 regions.
  unsigned n_jobs;

  ExecutionPlan execution_plan;

  size_t stack_size;  // The amount of stack required to run n_threads-many
                      // thread workers
  int stack_scratch_index;  // The buffer index where the above stack will be
                            // allocated

  // TODO: remove this  when better external memory handling is implemented
  // for loading from external mem
  int weights_scratch_idx = 0;
  int threshold_scratch_idx = 0;
  int bias_scratch_idx = 0;
  int multiplier_scratch_idx = 0;
  int accu_modifier_scratch_idx = 0;
  int output_trf_scratch_idx = 0;
};

// -------------------------------------------------------------------- //
// kernel types
// -------------------------------------------------------------------- //

enum class BConv2DKernelType {
  BITPACKED,
  BITPACKED_DI,
  INT8,
  INT8_DIDO,
};

#define UNSUPPORTED_KERNEL_TYPE \
  TF_LITE_FATAL("Unsupported BConv2DKernelType value")

template <BConv2DKernelType kernel_type>
struct BConv2DKernel {
  static inline const thread_function_t get_worker() {
    if (kernel_type == BConv2DKernelType::BITPACKED) {
      return bconv2d_bitpacked_thread_worker;
    } else if (kernel_type == BConv2DKernelType::BITPACKED_DI) {
      return bconv2d_bitpacked_deepin_thread_worker;
    } else if (kernel_type == BConv2DKernelType::INT8) {
      return bconv2d_int8_thread_worker;
    } else if (kernel_type == BConv2DKernelType::INT8_DIDO) {
      return bconv2d_int8_deepin_deepout_thread_worker;
    } else {
      UNSUPPORTED_KERNEL_TYPE;
    }
  };
  static inline void calculate_worker_stack_size(size_t &stack_size) {
    if (kernel_type == BConv2DKernelType::BITPACKED) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size,
                                    bconv2d_bitpacked_thread_worker);
    } else if (kernel_type == BConv2DKernelType::BITPACKED_DI) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size,
                                    bconv2d_bitpacked_deepin_thread_worker);
    } else if (kernel_type == BConv2DKernelType::INT8) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, bconv2d_int8_thread_worker);
    } else if (kernel_type == BConv2DKernelType::INT8_DIDO) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size,
                                    bconv2d_int8_deepin_deepout_thread_worker);
    } else {
      UNSUPPORTED_KERNEL_TYPE;
    }
  };
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto *op_data = reinterpret_cast<BConv2DOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(BConv2DOpData)));

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  TFLITE_DCHECK(length > 0);  // in fact it must be at least 6x uint32_t big

  auto parser = CustomOptionParser(buffer, length);
  auto Kshape = parser.parseNamedCustomOption("K").AsVector();
  op_data->args.y.channels = Kshape[0].AsUInt32();
  op_data->args.k.shape.height = Kshape[1].AsUInt32();
  op_data->args.k.shape.width = Kshape[2].AsUInt32();
  op_data->args.x.channels = Kshape[3].AsUInt32();

  auto strides = parser.parseNamedCustomOption("stride").AsVector();
  op_data->args.k.stride.vertical = strides[0].AsInt32();
  op_data->args.k.stride.horizontal = strides[1].AsInt32();

  op_data->args.k.dilation.horizontal = 1;
  op_data->args.k.dilation.vertical = 1;

  op_data->n_threads = 1;
  op_data->n_jobs = 1;

  TFLITE_DCHECK(op_data->n_threads > 0);
  TFLITE_DCHECK(op_data->n_jobs > 0);
  TFLITE_DCHECK(op_data->args.k.stride.horizontal > 0);
  TFLITE_DCHECK(op_data->args.k.stride.vertical > 0);
  TFLITE_DCHECK(op_data->args.k.dilation.horizontal > 0);
  TFLITE_DCHECK(op_data->args.k.dilation.vertical > 0);

  // Allocate the jobs (one pointer per thread)
  op_data->threads =
      reinterpret_cast<BConv2DThreadData *>(context->AllocatePersistentBuffer(
          context, sizeof(BConv2DThreadData) * op_data->n_threads));

  // Allocate the jobs (one BConv2DJob per region)
  op_data->jobs =
      reinterpret_cast<BConv2DJob *>(context->AllocatePersistentBuffer(
          context, sizeof(BConv2DJob) * op_data->n_jobs));

  // TODO: this will need the parsed parallelizaiton plan when available
  auto &job = op_data->jobs[0];
  job.top = 0;
  job.left = 0;
  auto &td = op_data->threads[0];
  td.job = &job;
  td.args = &op_data->args;

  return op_data;
}

static inline TfLiteStatus request_scratch_if_needed(TfLiteContext *context,
                                                     const TfLiteTensor *tensor,
                                                     int &scratch_idx) {
  if (!is_ram_address((uintptr_t)tensor->data.data)) {
    return context->RequestScratchBufferInArena(context, tensor->bytes,
                                                &scratch_idx);
  }
  return kTfLiteOk;
}

TfLiteStatus PrepareCommon(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto *op_data = reinterpret_cast<BConv2DOpData *>(node->user_data);

  const TfLiteTensor *input = GetInput(context, node, 0);
  op_data->args.x.height = (uint32_t)input->dims->data[1];
  op_data->args.x.width = (uint32_t)input->dims->data[2];

  const TfLiteTensor *output = GetOutput(context, node, 0);
  op_data->args.y.height = (uint32_t)output->dims->data[1];
  op_data->args.y.width = (uint32_t)output->dims->data[2];

  // TODO: remove this when parallelization is done
  op_data->jobs[0].cols = op_data->args.y.width;
  op_data->jobs[0].rows = op_data->args.y.height;

  return kTfLiteOk;
}

template <BConv2DKernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_STATUS(PrepareCommon(context, node));

  auto *op_data = reinterpret_cast<BConv2DOpData *>(node->user_data);

  // TODO: fix this this when parallelization is done
  // allocate scratch buffers for input parameter tensors (if necessary)
  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 1), op_data->weights_scratch_idx));

  if (kernel_type == BConv2DKernelType::BITPACKED ||
      kernel_type == BConv2DKernelType::BITPACKED_DI) {  // output is bitpacked
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 2), op_data->threshold_scratch_idx));
  } else if (kernel_type == BConv2DKernelType::INT8 ||
             kernel_type == BConv2DKernelType::INT8_DIDO) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node),
                      (kernel_type == BConv2DKernelType::INT8) ? 6 : 5);
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 2), op_data->multiplier_scratch_idx));
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 3), op_data->bias_scratch_idx));
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 4), op_data->output_trf_scratch_idx));
    if (kernel_type == BConv2DKernelType::INT8) {
      TF_LITE_ENSURE_STATUS(
          request_scratch_if_needed(context, GetInput(context, node, 5),
                                    op_data->accu_modifier_scratch_idx));
    }
  } else {
    UNSUPPORTED_KERNEL_TYPE;
  }

  BConv2DKernel<kernel_type>::calculate_worker_stack_size(op_data->stack_size);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->n_threads,
      &op_data->stack_scratch_index));

  if (kernel_type == BConv2DKernelType::BITPACKED ||
      kernel_type == BConv2DKernelType::INT8) {
    int thread_scratch_size =
        4 * (op_data->args.k.shape.height * op_data->args.k.shape.width *
                 op_data->args.x.channels / XS1_ALL_BITS_SIZE +
             XS3_VPU_VREG_WIDTH_WORDS);
    for (int thread_idx = 0; thread_idx < op_data->n_threads; thread_idx++) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, thread_scratch_size,
          &op_data->threads[thread_idx].thread_scratch_idx));
    }
  }

  return kTfLiteOk;
}

template <typename T>
static inline TfLiteStatus fetch_scratch_if_needed(TfLiteContext *context,
                                                   T *&array,
                                                   const TfLiteTensor *tensor,
                                                   int scratch_idx) {
  if (scratch_idx >= 0) {
    array =
        static_cast<const T *>(context->GetScratchBuffer(context, scratch_idx));
    GetDispatcher()->FetchBuffer((int8_t **)&array,
                                 GetTensorData<int8_t>(tensor), tensor->bytes);
  } else {
    array = GetTensorData<T>(tensor);
  }
  TF_LITE_ENSURE(context, array);
  return kTfLiteOk;
}

template <BConv2DKernelType kernel_type>
TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = reinterpret_cast<BConv2DOpData *>(node->user_data);
  op_data->args.X = GetTensorData<bnn_b32_t>(GetInput(context, node, 0));

  TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(context, op_data->args.K,
                                                GetInput(context, node, 1),
                                                op_data->weights_scratch_idx));

  if (kernel_type == BConv2DKernelType::BITPACKED ||
      kernel_type == BConv2DKernelType::BITPACKED_DI) {
    op_data->args.Y_bitpacked =
        GetTensorData<bnn_b32_t>(GetOutput(context, node, 0));
    TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
        context, op_data->args.thresholds, GetInput(context, node, 2),
        op_data->threshold_scratch_idx));
  } else if (kernel_type == BConv2DKernelType::INT8 ||
             kernel_type == BConv2DKernelType::INT8_DIDO) {
    op_data->args.Y_int8 = GetTensorData<int8_t>(GetOutput(context, node, 0));
    TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
        context, op_data->args.post_act_mult, GetInput(context, node, 2),
        op_data->multiplier_scratch_idx));
    TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
        context, op_data->args.post_act_bias, GetInput(context, node, 3),
        op_data->bias_scratch_idx));
    TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
        context, op_data->args.output_trf_parameters,
        GetInput(context, node, 4), op_data->output_trf_scratch_idx));

    if (kernel_type == BConv2DKernelType::INT8) {
      TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
          context, op_data->args.accu_modifier, GetInput(context, node, 5),
          op_data->accu_modifier_scratch_idx));
    }
  } else {
    UNSUPPORTED_KERNEL_TYPE;
  }

  // initialize the threads
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);

  Dispatcher *dispatcher = GetDispatcher();
  dispatcher->InitializeTasks(BConv2DKernel<kernel_type>::get_worker(), stack,
                              op_data->stack_size);

  // add tasks
  for (int thread_idx = 0; thread_idx < op_data->n_threads; thread_idx++) {
    auto &thread = op_data->threads[thread_idx];
    if (kernel_type == BConv2DKernelType::BITPACKED ||
        kernel_type == BConv2DKernelType::INT8) {
      thread.thread_scratch = static_cast<bnn_b32_t *>(
          context->GetScratchBuffer(context, thread.thread_scratch_idx));
    }
    dispatcher->AddTask(reinterpret_cast<void *>(&thread));
  }

  // start and wait for tasks to complete
  dispatcher->JoinTasks();

  return kTfLiteOk;
}

}  // namespace bconv

TfLiteRegistration *Register_BConv2D_Bitpacked_Deepin() {
  static TfLiteRegistration r = {
      bconv::Init, nullptr,
      bconv::Prepare<bconv::BConv2DKernelType::BITPACKED_DI>,
      bconv::Eval<bconv::BConv2DKernelType::BITPACKED_DI>};
  return &r;
}

TfLiteRegistration *Register_BConv2D_Bitpacked() {
  static TfLiteRegistration r = {
      bconv::Init, nullptr, bconv::Prepare<bconv::BConv2DKernelType::BITPACKED>,
      bconv::Eval<bconv::BConv2DKernelType::BITPACKED>};
  return &r;
}

TfLiteRegistration *Register_BConv2D_Int8_Deepin_Deepout() {
  static TfLiteRegistration r = {
      bconv::Init, nullptr, bconv::Prepare<bconv::BConv2DKernelType::INT8_DIDO>,
      bconv::Eval<bconv::BConv2DKernelType::INT8_DIDO>};
  return &r;
}

TfLiteRegistration *Register_BConv2D_Int8() {
  static TfLiteRegistration r = {bconv::Init, nullptr,
                                 bconv::Prepare<bconv::BConv2DKernelType::INT8>,
                                 bconv::Eval<bconv::BConv2DKernelType::INT8>};
  return &r;
}

}  // namespace xcore

}  // namespace micro
}  // namespace ops
}  // namespace tflite
