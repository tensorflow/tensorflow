#define SYSC

#include "tensorflow/lite/delegates/utils/vta_sim_delegate/vta_sim_delegate.h"
#include <utility>
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include "tensorflow/lite/delegates/utils/vta_sim_delegate/accelerator/driver/vta_gemm_driver.h"
#include "tensorflow/lite/delegates/utils/vta_sim_delegate/util.h"

#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/systemc_integrate.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"

// Some variables needs to be defined across multiple instances of the delegate
ACCNAME* acc;
struct del_params dparams;
struct sysC_sigs* scs;
struct Profile* profile;

bool save = true;
unsigned int vta_count = 0;
int cweight_offset = 0;

namespace tflite {
namespace vta_sim_test {

// VtaSim delegate kernel.
class VtaSimDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit VtaSimDelegateKernel(const VtaSimDelegateOptions& options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {

    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      static struct sysC_sigs _scs(1);
      static ACCNAME accelerator("accelerator");
      static struct Profile _profile;
      _profile.addMetric(DataCount("total_instruction_count"));
      sysC_init();
      systemC_binder(&accelerator, &_scs);
      acc = &accelerator;
      scs = &_scs;
      profile = &_profile;
      dparams.init = true;

      std::cout << "===========================" << std::endl;
      std::cout << "Initialised the SystemC Modules" << std::endl;
      std::cout << "VTA Accelerator";
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
    wt_sum.resize(params->nodes_to_replace->size);
    biases.resize(params->nodes_to_replace->size);
    crf.resize(params->nodes_to_replace->size);
    crx.resize(params->nodes_to_replace->size);
    weight_offsets.resize(params->nodes_to_replace->size);
    del_weights.resize(params->nodes_to_replace->size);

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
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

      TfLiteConvParams* cparam =
          reinterpret_cast<TfLiteConvParams*>(delegated_node->builtin_data);
      OpData* opdata = reinterpret_cast<OpData*>(delegated_node->user_data);

      cparams[i] = cparam;
      opdatas[i] = opdata;
    }
    return kTfLiteOk;
  }

  // Runs once per node before inference/invoke()
  // This function preloads weights, allocates additional tensors, calculates
  // quantization parameters For more info look into
  // "tensorflow/lite/kernels/conv.cc" for the default implementation for Conv2D
  // Nodes
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    KernelType kernel_type = kCblasOptimized;
    int node_count = inputs_.size();
    int out_tid = 0;

    for (int i = 0; i < node_count; i++) {
      TfLiteConvParams* params = cparams[i];
      OpData* data = opdatas[i];

      TfLiteTensor* output;
      const TfLiteTensor* input;
      const TfLiteTensor* filter;
      const TfLiteTensor* bias;

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
      const auto* affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization*>(
              filter->quantization.params);
      TF_LITE_ENSURE(context, affine_quantization);
      TF_LITE_ENSURE(context, affine_quantization->scale);
      TF_LITE_ENSURE(context,
                     (affine_quantization->scale->size == 1 ||
                      affine_quantization->scale->size == channels_out));

      data->per_channel_output_shift.resize(channels_out);
      crf[i] = new int[channels_out];
      crx[i] = new int8_t[channels_out];

      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
          context, input, filter, bias, output, params->activation,
          &data->output_multiplier, &data->output_shift,
          &data->output_activation_min, &data->output_activation_max,
          &crf[i][0], data->per_channel_output_shift.data(), channels_out));

      for (int j = 0; j < channels_out; j++)
        crx[i][j] = (int8_t)data->per_channel_output_shift.data()[j];

      TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
      output_size->data[0] = batches;
      output_size->data[1] = out_height;
      output_size->data[2] = out_width;
      output_size->data[3] = channels_out;
      auto output_status = context->ResizeTensor(context, output, output_size);
      if (output_status != kTfLiteOk) return output_status;

      if (data->need_im2col) {
        node->temporaries->data[data->im2col_index] = data->im2col_id;
        TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(4);
        int input_depth = input->dims->data[3];
        im2col_size->data[0] = output_size->data[0];
        im2col_size->data[1] = output_size->data[1];
        im2col_size->data[2] = output_size->data[2];
        im2col_size->data[3] = input_depth * filter_height * filter_width;

        TfLiteTensor* im2col =
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
        TfLiteIntArray* hwcn_weights_size = TfLiteIntArrayCreate(2);

        int input_depth = input->dims->data[3];
        hwcn_weights_size->data[0] =
            (filter_height * filter_width * input_depth);
        hwcn_weights_size->data[1] = channels_out;

        TfLiteTensor* hwcn_weights =
            &context
                 ->tensors[node->temporaries->data[data->hwcn_weights_index]];
        hwcn_weights->type = input->type;
        hwcn_weights->allocation_type = kTfLiteArenaRwPersistent;
        auto hwcn_weights_status =
            context->ResizeTensor(context, hwcn_weights, hwcn_weights_size);
        if (hwcn_weights_status != kTfLiteOk) return hwcn_weights_status;

        data->have_weights_been_transposed = false;
      }

      if (req_temp_out) {
        node->temporaries->data[temp_out_id] = outputs_[i][0];

        TfLiteIntArray* temp_out_tensor_size = TfLiteIntArrayCreate(4);
        temp_out_tensor_size->data[0] = output_size->data[0];
        temp_out_tensor_size->data[1] = output_size->data[1];
        temp_out_tensor_size->data[2] = output_size->data[2];
        temp_out_tensor_size->data[3] = output_size->data[3];

        TfLiteTensor* temp_out_tensor = &context->tensors[outputs_[i][0]];
        temp_out_tensor->type = kTfLiteInt8;
        temp_out_tensor->allocation_type = kTfLiteArenaRw;
        auto temp_out_tensor_status = context->ResizeTensor(
            context, temp_out_tensor, temp_out_tensor_size);
        if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
      }

      // Might need to transform this in a bias matrix
      biases[i] = bias->data.i32;
      int* dims = filter->dims->data;
      int* in_dims = input->dims->data;
      preload_weights(filter->data.int8, dims, wt_sum[i]);

      const int8* filter_data = filter->data.int8;
      int M = dims[0];
      int K = dims[1] * dims[2] * dims[3];
      int N = out_height * out_width * batches;
      int pN = roundUp(N, 16);
      int pM = roundUp(M, 16);
      int pK = roundUp(K, 16);

      int8_t** padded_weights = new int8_t*[pM]();
      for (int j = 0; j < pM; ++j) padded_weights[j] = new int8_t[pK]();
      pad_matrix(M, K, 16, 16, filter_data, padded_weights);

      uint32_t* weight_buf = new uint32_t[pK * pM];
      int block_factor = VTA_BLOCK_OUT;
      tflite_vtasim::packBufferBlock<uint32_t, 32, int8_t, 8>(
          weight_buf, padded_weights, 0, pM, block_factor, 0, pK, VTA_BLOCK_IN);
      del_weights[i] = weight_buf;
      cweight_offset += pM * pK / 4;
    }
    return kTfLiteOk;
  }

  // Runs once per node during inference/invoke()
  // This function executes the operations required by node by offloading the
  // computation to the gemm_driver For more info look into
  // "tensorflow/lite/kernels/conv.cc" for the default implementation for Conv2D
  // Nodes
  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      auto* params = cparams[i];
      OpData* data = opdatas[i];

      const TfLiteTensor* input;
      const TfLiteTensor* filter;
      TfLiteTensor* output;
      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      GetOutputSafe(context, outputs_[i][0], &output);

      TfLiteTensor* im2col =
          data->need_im2col
              ? &context->tensors[node->temporaries->data[data->im2col_index]]
              : nullptr;

      const int8* input_data = input->data.int8;
      const int8* filter_data = filter->data.int8;
      int8* im2col_data = data->need_im2col ? im2col->data.int8 : nullptr;
      int8* output_data = output->data.int8;

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

      const int8* gemm_input_data = nullptr;
      const RuntimeShape* gemm_input_shape = nullptr;
      const int filter_width = filter_shape.Dims(2);
      const int filter_height = filter_shape.Dims(1);
      const bool need_dilated_im2col =
          dilation_width_factor != 1 || dilation_height_factor != 1;
      const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                               filter_width != 1 || filter_height != 1;
      const int8 input_zero_point = -input_offset;
      const uint8 zero_point_byte =
          *reinterpret_cast<const uint8*>(&input_zero_point);
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

      //  Load & Reshape Input data to temporary buffers before offloading to
      //  DMA inbuffers
      int width = gemm_input_cols;
      int w = ((width + 3) - ((width + 3) % 4));
      int depth = filter_cols;
      int d = ((depth + 15) - ((depth + 15) % 16));
      int d2 = depth * 2;
      int d3 = depth * 3;
      int d4 = depth * 4;
      int i_c = 0;
      int sums_curr = 0;
      int in_sum[w];

      int dm = 0;
      for (int i = 0; i < w / 4; i++) {
        int id = i * d4;
        int i0 = id;
        int i1 = id + depth;
        int i2 = id + d2;
        int i3 = id + d3;
        int ss0 = 0;
        int ss1 = 0;
        int ss2 = 0;
        int ss3 = 0;

        for (int j = dm; j < d; j++) {
          if (j < depth) {
            unsigned char w0 = gemm_input_data[i0 + j];
            unsigned char w1 = gemm_input_data[i1 + j];
            unsigned char w2 = gemm_input_data[i2 + j];
            unsigned char w3 = gemm_input_data[i3 + j];
            ss0 += w0;
            ss1 += w1;
            ss2 += w2;
            ss3 += w3;
          }
        }
        in_sum[sums_curr++] = (ss0);
        in_sum[sums_curr++] = (ss1);
        in_sum[sums_curr++] = (ss2);
        in_sum[sums_curr++] = (ss3);
      }

      int N = gemm_input_cols;
      int M = output_rows;  
      int K = gemm_input_rows; 

      int rfactor = 16;
      int pN = roundUp(N, rfactor);
      int pM = roundUp(M, rfactor);
      int pK = roundUp(K, rfactor);

      int8_t** padded_input = new int8_t*[pN];
      for (int i = 0; i < pN; ++i) padded_input[i] = new int8_t[pK];

      int8_t** padded_weights = new int8_t*[pM];
      for (int i = 0; i < pM; ++i) padded_weights[i] = new int8_t[pK];

      bool flipped = false;
      int oN = flipped ? pM : pN;
      int oM = flipped ? pN : pM;

      int8_t** padded_output = new int8_t*[oN];
      for (int i = 0; i < oN; ++i) padded_output[i] = new int8_t[oM];

      pad_matrix(N, K, 16, 16, gemm_input_data, padded_input);
      pad_matrix(M, K, 16, 16, filter_data, padded_weights);

      // acc_container is used to wrap all the paramters the
      // vta_gemm_driver/accelerator needs from the delegate
      struct acc_container drv(padded_input, padded_weights, padded_output,
                               wt_sum[i], crf[i], crx[i]);
      drv.scs = scs;
      drv.profile = profile;
      drv.acc = acc;
      drv.bias = biases[i];
      drv.in_sum = in_sum;
      drv.ra = output_offset;
      drv.rhs_offset = input_offset;
      drv.lhs_offset = 0;
      drv.pN = pN;
      drv.pM = pM;
      drv.pK = pK;
      drv.N = N;
      drv.M = M;
      drv.K = K;
      drv.packed_weights = del_weights[i];
      drv.flipped = flipped;
      acc->layer =  dparams.layer;
      drv.layer =  dparams.layer;
      drv.save = save;
      drv.vta_count = vta_count;
      tflite_vtasim::Entry(drv);
      vta_count = drv.vta_count;
      save = drv.save;

      if (flipped)
        unpadT_matrix(N, M, 16, 16, padded_output, output_data);
      else
        unpad_matrix(N, M, 16, 16, padded_output, output_data);

      dparams.layer++;
      dparams.delegated_nodes--;
    }

    if (dparams.delegated_nodes == 0) {
      profile->saveCSVRecords("z_VTA_Sim");
    }
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;
  std::vector<std::vector<int>> wt_sum;
  std::vector<int> weight_offsets;
  std::vector<uint32_t*> del_weights;
  std::vector<int*> biases;
  std::vector<int*> crf;
  std::vector<int8_t*> crx;
  std::vector<OpData*> opdatas;
  std::vector<TfLiteConvParams*> cparams;

 private:
  const VtaSimDelegateOptions options_;
};

// VtaSimDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class VtaSimDelegate : public SimpleDelegateInterface {
 public:
  explicit VtaSimDelegate(const VtaSimDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports CONV2D op
    if (kTfLiteBuiltinConv2d != registration->builtin_code) return false;

    // This delegate only supports int8 types.
    if (node->inputs->size != 3) return false;

    for (int i = 0; i < 2; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.dims->data[0] == 1 && tensor.dims->data[1] == 1 &&
          tensor.dims->data[2] == 1) {
        return false;
      }
      if (tensor.type != kTfLiteInt8) return false;
    }
    auto& tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32) return false;

    // CONV2D
    dparams.delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "VtaSimDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<VtaSimDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const VtaSimDelegateOptions options_;
};

}  // namespace vta_sim_test
}  // namespace tflite

VtaSimDelegateOptions TfLiteVtaSimDelegateOptionsDefault() {
  VtaSimDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this vta_sim test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteVtaSimDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteVtaSimDelegateCreate(
    const VtaSimDelegateOptions* options) {
  std::unique_ptr<tflite::vta_sim_test::VtaSimDelegate> vta_sim(
      new tflite::vta_sim_test::VtaSimDelegate(
          options ? *options : TfLiteVtaSimDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(vta_sim));
}

// Destroys a delegate created with `TfLiteVtaSimDelegateCreate` call.
void TfLiteVtaSimDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
