

#include "tensorflow/lite/delegates/utils/secda_vm_delegate/secda_vm_delegate.h"

#include <utility>

#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include "tensorflow/lite/delegates/utils/secda_vm_delegate/driver/gemm_driver.h"
#include "tensorflow/lite/delegates/utils/secda_vm_delegate/util.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#define DMA_BC 10
// DRIVER Globals
static unsigned int dma_addrs[4] = {dma_addr0, dma_addr1, dma_addr2, dma_addr3};
static unsigned int dma_addrs_in[4] = {dma_in0, dma_in1, dma_in2, dma_in3};
static unsigned int dma_addrs_out[4] = {dma_out0, dma_out1, dma_out2, dma_out3};
static struct multi_dma mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out, DMA_BL);
static struct del_params dparams;
static struct vm_times vm_t;
struct store_params st_params[DMA_BC];
struct dma_buffer_set dfs[4] = {
    {DMA_BC, 204800, dma_in0},
    {DMA_BC, 204800, dma_in1},
    {DMA_BC, 204800, dma_in2},
    {DMA_BC, 204800, dma_in3},
};
int recv_len = 204800 / DMA_BC;

namespace tflite {
namespace secda_vm_test {

// SecdaVM delegate kernel.
class SecdaVMDelegateKernel : public SimpleDelegateKernelInterface {
public:
  explicit SecdaVMDelegateKernel(const SecdaVMDelegateOptions &options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext *context,
                    const TfLiteDelegateParams *params) override {
    // Init DMA
    if (!dparams.init) {
      dparams.acc = getAccBaseAddress<int>(acc_address, 65536);
      dparams.init = true;
      std::cout << "===========================" << std::endl;
      std::cout << "Initialised the DMA" << std::endl;
      std::cout << "Vector MAC";
#ifdef ACC_NEON
      std::cout << " with Neon";
#endif
      std::cout << std::endl;
      std::cout << "===========================" << std::endl;
    }

    // Save Tensors input & outputs
    // Save other info (opdata)
    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    opdatas.resize(params->nodes_to_replace->size);
    cparams.resize(params->nodes_to_replace->size);
    biases.resize(params->nodes_to_replace->size);
    crf.resize(params->nodes_to_replace->size);
    crx.resize(params->nodes_to_replace->size);
    wb0.resize(params->nodes_to_replace->size);
    wb1.resize(params->nodes_to_replace->size);
    wb2.resize(params->nodes_to_replace->size);
    wb3.resize(params->nodes_to_replace->size);
    wt_sum1.resize(params->nodes_to_replace->size);
    wt_sum2.resize(params->nodes_to_replace->size);
    wt_sum3.resize(params->nodes_to_replace->size);
    wt_sum4.resize(params->nodes_to_replace->size);

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode *delegated_node = nullptr;
      TfLiteRegistration *delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      inputs_[i].push_back(delegated_node->inputs->data[2]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
      associated_nodes.push_back(node_index);
      TfLiteConvParams *cparam =
          reinterpret_cast<TfLiteConvParams *>(delegated_node->builtin_data);
      OpData *opdata = reinterpret_cast<OpData *>(delegated_node->user_data);
      cparams[i] = cparam;
      opdatas[i] = opdata;
    }
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) override {
    KernelType kernel_type = kCblasOptimized;
    int node_count = inputs_.size();
    int out_tid = 0;
    for (int i = 0; i < node_count; i++) {
      TfLiteConvParams *params = cparams[i];
      OpData *data = opdatas[i];

      TfLiteTensor *output;
      const TfLiteTensor *input;
      const TfLiteTensor *filter;
      const TfLiteTensor *bias;

      GetOutputSafe(context, outputs_[i][0], &output);
      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      GetInputSafe(context, inputs_[i][2], &bias);

      const bool is_hybrid = false;
      int channels_in = filter->dims->data[3];
      int channels_out = filter->dims->data[0];
      int width = input->dims->data[2];
      int height = input->dims->data[1];
      int filter_width = filter->dims->data[2];
      int filter_height = filter->dims->data[1];
      int batches = input->dims->data[0];

      auto padding = params->padding;
      int out_width, out_height;
      data->padding = ComputePaddingHeightWidth(
          params->stride_height, params->stride_width,
          params->dilation_height_factor, params->dilation_width_factor, height,
          width, filter_height, filter_width, padding, &out_height, &out_width);

      size_t im2col_type_size = sizeof(int8_t);
      const size_t im2col_bytes = static_cast<size_t>(batches) * out_height *
                                  out_width * channels_in * filter_height *
                                  filter_width * im2col_type_size;

      int temp_out_id;
      bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
      if (!req_temp_out) out_tid++;

      TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
          context, node, is_hybrid, data->is_hybrid_per_channel, kernel_type,
          im2col_bytes, params, data, req_temp_out, outputs_[i][0], temp_out_id,
          inputs_[i][0], inputs_[i][1]));

      TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                        kTfLiteAffineQuantization);
      const auto *affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization *>(
              filter->quantization.params);
      TF_LITE_ENSURE(context, affine_quantization);
      TF_LITE_ENSURE(context, affine_quantization->scale);
      TF_LITE_ENSURE(context,
                     (affine_quantization->scale->size == 1 ||
                      affine_quantization->scale->size == channels_out));

      data->per_channel_output_shift.resize(channels_out);
      crf[i].resize(channels_out);
      crx[i].resize(channels_out);

      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
          context, input, filter, bias, output, params->activation,
          &data->output_multiplier, &data->output_shift,
          &data->output_activation_min, &data->output_activation_max,
          &crf[i][0], data->per_channel_output_shift.data(), channels_out));

      for (int j = 0; j < channels_out; j++)
        crx[i][j] = (int8_t)data->per_channel_output_shift.data()[j];

      TfLiteIntArray *output_size = TfLiteIntArrayCreate(4);
      output_size->data[0] = batches;
      output_size->data[1] = out_height;
      output_size->data[2] = out_width;
      output_size->data[3] = channels_out;
      auto output_status = context->ResizeTensor(context, output, output_size);
      if (output_status != kTfLiteOk) return output_status;

      if (data->need_im2col) {
        node->temporaries->data[data->im2col_index] = data->im2col_id;
        TfLiteIntArray *im2col_size = TfLiteIntArrayCreate(4);
        int input_depth = input->dims->data[3];
        im2col_size->data[0] = output_size->data[0];
        im2col_size->data[1] = output_size->data[1];
        im2col_size->data[2] = output_size->data[2];
        im2col_size->data[3] = input_depth * filter_height * filter_width;

        TfLiteTensor *im2col =
            &context->tensors[node->temporaries->data[data->im2col_index]];
        im2col->type = input->type;
        if (is_hybrid) {
          im2col->type = filter->type;
        }
        im2col->allocation_type = kTfLiteArenaRw;
        auto im2col_status =
            context->ResizeTensor(context, im2col, im2col_size);
        if (im2col_status != kTfLiteOk) return im2col_status;
      }

      if (data->need_hwcn_weights) {
        node->temporaries->data[data->hwcn_weights_index] =
            data->hwcn_weights_id;
        TfLiteIntArray *hwcn_weights_size = TfLiteIntArrayCreate(2);

        int input_depth = input->dims->data[3];
        hwcn_weights_size->data[0] =
            (filter_height * filter_width * input_depth);
        hwcn_weights_size->data[1] = channels_out;

        TfLiteTensor *hwcn_weights =
            &context
                 ->tensors[node->temporaries->data[data->hwcn_weights_index]];
        hwcn_weights->type = input->type;
        hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;
        auto hwcn_weights_status =
            context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
        if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

        data->have_weights_been_transposed = false;
      }

      if (data->need_im2col) {
        node->temporaries->data[data->im2col_index] = data->im2col_id;
        TfLiteIntArray *im2col_size = TfLiteIntArrayCreate(4);
        int input_depth = input->dims->data[3];
        im2col_size->data[0] = output_size->data[0];
        im2col_size->data[1] = output_size->data[1];
        im2col_size->data[2] = output_size->data[2];
        im2col_size->data[3] = input_depth * filter_height * filter_width;

        TfLiteTensor *im2col =
            &context->tensors[node->temporaries->data[data->im2col_index]];
        im2col->type = input->type;
        if (is_hybrid) {
          im2col->type = filter->type;
        }
        im2col->allocation_type = kTfLiteArenaRw;
        auto im2col_status =
            context->ResizeTensor(context, im2col, im2col_size);
        if (im2col_status != kTfLiteOk) return im2col_status;
      }

      if (req_temp_out) {
        node->temporaries->data[temp_out_id] = outputs_[i][0];

        TfLiteIntArray *temp_out_tensor_size = TfLiteIntArrayCreate(4);
        temp_out_tensor_size->data[0] = output_size->data[0];
        temp_out_tensor_size->data[1] = output_size->data[1];
        temp_out_tensor_size->data[2] = output_size->data[2];
        temp_out_tensor_size->data[3] = output_size->data[3];

        TfLiteTensor *temp_out_tensor = &context->tensors[outputs_[i][0]];
        temp_out_tensor->type = kTfLiteInt8;
        temp_out_tensor->allocation_type = kTfLiteArenaRw;
        auto temp_out_tensor_status = context->ResizeTensor(
            context, temp_out_tensor, temp_out_tensor_size);
        if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
      }

      biases[i] = bias->data.i32;
      int *dims = filter->dims->data;
      preload_weights(filter->data.int8, dims, wb0[i], wb1[i], wb2[i], wb3[i],
                      wt_sum1[i], wt_sum2[i], wt_sum3[i], wt_sum4[i]);
    }

    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) override {
    prf_start(0);
    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      prf_start(1);
      auto *params = cparams[i];
      OpData *data = opdatas[i];
      const TfLiteTensor *input;
      const TfLiteTensor *filter;
      TfLiteTensor *output;

      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      GetOutputSafe(context, outputs_[i][0], &output);

      TfLiteTensor *im2col =
          data->need_im2col
              ? &context->tensors[node->temporaries->data[data->im2col_index]]
              : nullptr;

      const int8 *input_data = input->data.int8;
      const int8 *filter_data = filter->data.int8;
      int8 *im2col_data = data->need_im2col ? im2col->data.int8 : nullptr;
      int8 *output_data = output->data.int8;

      ConvParams op_params;
      op_params.input_offset = -input->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.stride_height = params->stride_height;
      op_params.stride_width = params->stride_width;
      op_params.dilation_height_factor = params->dilation_height_factor;
      op_params.dilation_width_factor = params->dilation_width_factor;
      op_params.padding_values.height = data->padding.height;
      op_params.padding_values.width = data->padding.width;
      op_params.quantized_activation_min = data->output_activation_min;
      op_params.quantized_activation_max = data->output_activation_max;

      RuntimeShape input_shape =
          RuntimeShape(input->dims->size, input->dims->data);
      RuntimeShape filter_shape =
          RuntimeShape(filter->dims->size, filter->dims->data);
      RuntimeShape output_shape =
          RuntimeShape(output->dims->size, output->dims->data);

      const int stride_width = params->stride_width;
      const int stride_height = params->stride_height;

      const int dilation_width_factor = params->dilation_width_factor;
      const int dilation_height_factor = params->dilation_height_factor;
      const int32 input_offset = -input->params.zero_point;
      const int32 output_offset = output->params.zero_point;
      // Set min and max value of the output.
      const int32 output_activation_min = data->output_activation_min;
      const int32 output_activation_max = data->output_activation_max;

      const int8 *gemm_input_data = nullptr;
      const RuntimeShape *gemm_input_shape = nullptr;
      const int filter_width = filter_shape.Dims(2);
      const int filter_height = filter_shape.Dims(1);
      const bool need_dilated_im2col =
          dilation_width_factor != 1 || dilation_height_factor != 1;
      const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                               filter_width != 1 || filter_height != 1;
      const int8 input_zero_point = -input_offset;
      const uint8 zero_point_byte =
          *reinterpret_cast<const uint8 *>(&input_zero_point);
      if (need_dilated_im2col) {
        TFLITE_DCHECK(im2col_data);
        RuntimeShape im2col_shape =
            RuntimeShape(im2col->dims->size, im2col->dims->data);
        DilatedIm2col<int8_t>(op_params, zero_point_byte, input_shape,
                              input_data, filter_shape, output_shape,
                              im2col_data);

        gemm_input_data = im2col_data;
        gemm_input_shape = &im2col_shape;
      } else if (need_im2col) {
        TFLITE_DCHECK(im2col_data);
        RuntimeShape im2col_shape =
            RuntimeShape(im2col->dims->size, im2col->dims->data);
        Im2col<int8_t>(op_params, filter_height, filter_width, zero_point_byte,
                       input_shape, input_data, im2col_shape, im2col_data);
        gemm_input_data = im2col_data;
        gemm_input_shape = &im2col_shape;
      } else {
        TFLITE_DCHECK(!im2col_data);
        gemm_input_data = input_data;
        gemm_input_shape = &input_shape;
      }

      const int gemm_input_rows = gemm_input_shape->Dims(3);
      const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
      const int filter_rows = filter_shape.Dims(0);
      const int filter_cols = FlatSizeSkipDim(filter_shape, 0);
      const int output_rows = output_shape.Dims(3);
      const int output_cols =
          output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);

      int width = gemm_input_cols;
      int w = ((width + 3) - ((width + 3) % 4));
      int depth = filter_cols;
      int d = ((depth + 15) - ((depth + 15) % 16));
      int s_need = w * d / 4 + 1;
      int in_sum1[w / 4];
      int in_sum2[w / 4];
      int in_sum3[w / 4];
      int in_sum4[w / 4];
      int8_t inb0[s_need];
      int8_t inb1[s_need];
      int8_t inb2[s_need];
      int8_t inb3[s_need];
      precal_sum_load_pad(gemm_input_data, width, depth, inb0, inb1, inb2, inb3,
                          in_sum1, in_sum2, in_sum3, in_sum4);

      int *wb_0 = reinterpret_cast<int *>(&wb0[i][0]);
      int *wb_1 = reinterpret_cast<int *>(&wb1[i][0]);
      int *wb_2 = reinterpret_cast<int *>(&wb2[i][0]);
      int *wb_3 = reinterpret_cast<int *>(&wb3[i][0]);

      struct acc_container drv(wb_0, wb_1, wb_2, wb_3, wt_sum1[i], wt_sum2[i],
                               wt_sum3[i], wt_sum4[i], crf[i], crx[i]);
      drv.mdma = &mdma;
      drv.st_params = st_params;
      drv.dfs = dfs;
      drv.mt_context = &dparams.mt_context;
      drv.thread_count = context->recommended_num_threads;
      drv.in_id = 0;
      int *inb_0 = reinterpret_cast<int *>(inb0);
      int *inb_1 = reinterpret_cast<int *>(inb1);
      int *inb_2 = reinterpret_cast<int *>(inb2);
      int *inb_3 = reinterpret_cast<int *>(inb3);
      drv.inb_0 = inb_0;
      drv.inb_1 = inb_1;
      drv.inb_2 = inb_2;
      drv.inb_3 = inb_3;
      drv.in_sum1 = in_sum1;
      drv.in_sum2 = in_sum2;
      drv.in_sum3 = in_sum3;
      drv.in_sum4 = in_sum4;
      drv.bias = biases[i];
      drv.ra = output_offset;
      drv.rhs_offset = input_offset;
      drv.lhs_offset = 0;
      drv.t.layer = associated_nodes[i];
      drv.recv_len = recv_len;

      drv.rows = gemm_input_cols;
      drv.cols = filter_rows;
      drv.depth = filter_cols;

      int8_params ts_lhs_params;
      int8_params ts_rhs_params;
      int8_params ts_dst_params;
      ts_lhs_params.Init(filter_data, 0, filter_rows, filter_cols, filter_cols,
                         0);
      ts_rhs_params.Init(gemm_input_data, 1, gemm_input_rows, gemm_input_cols,
                         gemm_input_rows, 0);
      ts_dst_params.Init(output_data, 1, output_rows, output_cols, 0);

#ifdef DELEGATE_VERBOSE
      cout << "===========================" << endl;
      cout << "Layer: " << dparams.layer
           << "      Node: " << associated_nodes[i] << endl;
      cout << "===========================" << endl;
#endif

      prf_end(1, vm_t.ipack);
      drv.t2 = vm_t;
      // tflite_secda_vm::Entry(drv, ts_lhs_params, ts_rhs_params,
      // ts_dst_params);
      tflite_secda_vm::Entry(drv, output_data);
      vm_t = drv.t2;
      dparams.layer++;
      dparams.delegated_nodes--;
    }

    prf_end(0, vm_t.conv_total);
    if (dparams.delegated_nodes == 0) vm_t.print();
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<std::vector<int>> wt_sum1;
  std::vector<std::vector<int>> wt_sum2;
  std::vector<std::vector<int>> wt_sum3;
  std::vector<std::vector<int>> wt_sum4;
  std::vector<std::vector<int8_t>> wb0;
  std::vector<std::vector<int8_t>> wb1;
  std::vector<std::vector<int8_t>> wb2;
  std::vector<std::vector<int8_t>> wb3;
  std::vector<int *> biases;
  std::vector<std::vector<int>> crf;
  std::vector<std::vector<int8_t>> crx;
  std::vector<OpData *> opdatas;
  std::vector<TfLiteConvParams *> cparams;

private:
  const SecdaVMDelegateOptions options_;
};

// SecdaVMDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class SecdaVMDelegate : public SimpleDelegateInterface {
public:
  explicit SecdaVMDelegate(const SecdaVMDelegateOptions &options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration *registration,
                                 const TfLiteNode *node,
                                 TfLiteContext *context) const override {
    // Only supports CONV2D op
    if (kTfLiteBuiltinConv2d != registration->builtin_code) return false;

    // This delegate only supports int8 types
    if (node->inputs->size != 3) return false;
    for (int i = 0; i < 2; ++i) {
      auto &tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    // Ensures bias tensor is supports int32 type
    auto &tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32) return false;

    // Adds node for delegation
    dparams.delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext *context) override { return kTfLiteOk; }

  const char *Name() const override {
    static constexpr char kName[] = "SecdaVMDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<SecdaVMDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

private:
  const SecdaVMDelegateOptions options_;
};

} // namespace secda_vm_test
} // namespace tflite

SecdaVMDelegateOptions TfLiteSecdaVMDelegateOptionsDefault() {
  SecdaVMDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this secda_vm test delegate
  // will not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteSecdaVMDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate *
TfLiteSecdaVMDelegateCreate(const SecdaVMDelegateOptions *options) {
  std::cout << "===========================" << std::endl;
  std::cout << "Created" << std::endl;
  std::cout << "===========================" << std::endl;
  std::unique_ptr<tflite::secda_vm_test::SecdaVMDelegate> secda_vm(
      new tflite::secda_vm_test::SecdaVMDelegate(
          options ? *options : TfLiteSecdaVMDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(secda_vm));
}

// Destroys a delegate created with `TfLiteSecdaVMDelegateCreate` call.
void TfLiteSecdaVMDelegateDelete(TfLiteDelegate *delegate) {
  if (!dparams.unmap) {
    mdma.multi_free_dmas();
    std::cout << "===========================" << std::endl;
    std::cout << "Unmapped DMA I/O Buffers" << std::endl;
    std::cout << "===========================" << std::endl;
    dparams.unmap = true;
    for (int i = 0; i < 4; i++)
      dfs[i].free();
  }
  std::cout << "===========================" << std::endl;
  std::cout << "Deleted" << std::endl;
  std::cout << "===========================" << std::endl;
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
