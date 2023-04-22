#define SYSC

#include "tensorflow/lite/delegates/utils/tconv_sim_delegate/tconv_sim_delegate.h"

#include <iomanip>
#include <iostream>
#include <utility>

#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/systemc_integrate.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/delegates/utils/tconv_sim_delegate/accelerator/driver/gemm_driver.h"
#include "tensorflow/lite/delegates/utils/tconv_sim_delegate/util.h"

// Some variables needs to be defined across multiple instances of the delegate
unsigned int dma_addrs[4] = {0, 0, 0, 0};
unsigned int dma_addrs_in[4] = {0, 0, 0, 0};
unsigned int dma_addrs_out[4] = {0, 0, 0, 0};
struct multi_dma* mdma;
ACCNAME* acc;
struct del_params dparams;
struct Profile* profile;

namespace tflite {
namespace tconvsim_test {

// TCONVSim delegate kernel
class TCONVSimDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit TCONVSimDelegateKernel(const TCONVSimDelegateOptions& options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // // Init SystemC Modules & Profilier
    if (!dparams.init) {
      static struct sysC_sigs scs1(1);
      static ACCNAME _acc("TCONV");
      static struct multi_dma _mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                                    563840);
      static struct Profile _profile;
      sysC_init();
      sysC_binder(&_acc, &_mdma, &scs1);
      mdma = &_mdma;
      acc = &_acc;
      profile = &_profile;
      dparams.init = true;

      std::cout << "===========================" << std::endl;
      std::cout << "Initialised the SystemC Modules" << std::endl;
      std::cout << "Vector MAC Accelerator";
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
    tparams.resize(params->nodes_to_replace->size);
    biases.resize(params->nodes_to_replace->size);
    crf.resize(params->nodes_to_replace->size);
    crx.resize(params->nodes_to_replace->size);
    crx_8.resize(params->nodes_to_replace->size);
    wb0.resize(params->nodes_to_replace->size);
    wb1.resize(params->nodes_to_replace->size);
    wb2.resize(params->nodes_to_replace->size);
    wb3.resize(params->nodes_to_replace->size);
    wt_sum1.resize(params->nodes_to_replace->size);
    wt_sum2.resize(params->nodes_to_replace->size);
    wt_sum3.resize(params->nodes_to_replace->size);
    wt_sum4.resize(params->nodes_to_replace->size);
    wt_sum4.resize(params->nodes_to_replace->size);

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
      inputs_[i].push_back(delegated_node->inputs->data[3]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
      associated_nodes.push_back(node_index);
      TfLiteTransposeConvParams* tparam =
          reinterpret_cast<TfLiteTransposeConvParams*>(
              delegated_node->builtin_data);
      OpData* opdata = reinterpret_cast<OpData*>(delegated_node->user_data);
      tparams[i] = tparam;
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
    int node_count = inputs_.size();
    int out_tid = 0;

    for (int i = 0; i < node_count; i++) {
      TfLiteTransposeConvParams* params = tparams[i];
      OpData* data = opdatas[i];

      TfLiteTensor* output;
      const TfLiteTensor* output_shape;
      const TfLiteTensor* input;
      const TfLiteTensor* weights;
      const TfLiteTensor* bias;

      GetOutputSafe(context, outputs_[i][0], &output);
      GetInputSafe(context, inputs_[i][0], &output_shape);
      GetInputSafe(context, inputs_[i][1], &weights);
      GetInputSafe(context, inputs_[i][2], &input);
      GetInputSafe(context, inputs_[i][3], &bias);

      if (SizeOfDimension(input, 3) != SizeOfDimension(weights, 3))
        return kTfLiteError;

      OpData* user_data = opdatas[i];
      int temp_out_id;
      bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
      if (!req_temp_out) out_tid++;

      TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
          context, node, params, data, req_temp_out, outputs_[i][0],
          temp_out_id));

      TfLiteTensor* col2im = nullptr;
      if (data->has_col2im) {
        node->temporaries->data[data->col2im_index] = data->col2im_id;
        TF_LITE_ENSURE_OK(
            context,
            GetTemporarySafe(context, node, user_data->col2im_index, &col2im));
      }

      if (!IsConstantTensor(output_shape)) {
        // Defer resizing until Eval().
        SetTensorToDynamic(output);
        if (data->has_col2im) {
          SetTensorToDynamic(col2im);
        }
      } else {
        TF_LITE_ENSURE_STATUS(ResizeTensor(context, output_shape, output));
        if (data->has_col2im) {
          TF_LITE_ENSURE_STATUS(ResizeCol2ImTensor(context, output_shape,
                                                   weights, input, col2im));
        }
      }

      if (req_temp_out && !IsConstantTensor(output_shape)) {
        node->temporaries->data[temp_out_id] = outputs_[i][0];
        TfLiteTensor* temp_out_tensor = &context->tensors[outputs_[i][0]];
        temp_out_tensor->type = kTfLiteInt8;
        temp_out_tensor->allocation_type = kTfLiteDynamic;
        TF_LITE_ENSURE_STATUS(
            ResizeTensor(context, output_shape, temp_out_tensor));
      }

      if (data->weights_are_transposed) {
        node->temporaries->data[data->transposed_weights_index] =
            data->transposed_weights_id;
        TfLiteTensor* transposed_weights;
        TF_LITE_ENSURE_OK(
            context,
            GetTemporarySafe(context, node, user_data->transposed_weights_index,
                             &transposed_weights));
        if (!IsConstantTensor(weights)) {
          SetTensorToDynamic(transposed_weights);
        } else {
          ResizeAndTransposeWeights(context, weights, transposed_weights);
        }
      }

      node->temporaries->data[data->scratch_tensor_index] =
          data->scratch_tensor_id;
      TfLiteTensor* scratch_buffer;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, data->scratch_tensor_index,
                                    &scratch_buffer));

      scratch_buffer->type = kTfLiteInt32;
      scratch_buffer->allocation_type = kTfLiteDynamic;
      if (!IsConstantTensor(output_shape)) {
        SetTensorToDynamic(scratch_buffer);
      } else {
        TF_LITE_ENSURE_STATUS(
            ResizeTensor(context, output_shape, scratch_buffer));
      }

      TF_LITE_ENSURE_EQ(context, weights->quantization.type,
                        kTfLiteAffineQuantization);
      const auto* affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization*>(
              weights->quantization.params);
      const int channels_out = weights->dims->data[0];
      TF_LITE_ENSURE(context, affine_quantization);
      TF_LITE_ENSURE(context, affine_quantization->scale);
      TF_LITE_ENSURE(context,
                     (affine_quantization->scale->size == 1 ||
                      affine_quantization->scale->size == channels_out));

      data->per_channel_output_multiplier.resize(channels_out);
      data->per_channel_output_shift.resize(channels_out);

      crf[i].resize(channels_out);
      crx[i].resize(channels_out);
      crx_8[i].resize(channels_out);
      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
          context, input, weights, bias, output, kTfLiteActNone,
          &data->output_multiplier, &data->output_shift,
          &data->output_activation_min, &data->output_activation_max,
          &crf[i][0], data->per_channel_output_shift.data(), channels_out));
      for (int j = 0; j < channels_out; j++) {
        crx[i][j] = data->per_channel_output_shift.data()[j];
        crx_8[i][j] = (int8_t)data->per_channel_output_shift.data()[j];
      }

      biases[i] = bias->data.i32;

      TfLiteTensor* transposed_weights;
      TF_LITE_ENSURE_OK(
          context,
          GetTemporarySafe(context, node, user_data->transposed_weights_index,
                           &transposed_weights));
      int* dims = transposed_weights->dims->data;
      preload_weights(transposed_weights->data.int8, dims, wb0[i], wb1[i],
                      wb2[i], wb3[i], wt_sum1[i], wt_sum2[i], wt_sum3[i],
                      wt_sum4[i]);

      int k = 0;
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
      auto* params = tparams[i];
      OpData* data = opdatas[i];
      const TfLiteTensor* input;
      const TfLiteTensor* weights;
      TfLiteTensor* output;
      const TfLiteTensor* bias;
      const TfLiteTensor* output_shape_tensor;
      bool has_bias = inputs_[i][3] != 0;

      TfLiteTensor* transposed_weights =
          data->weights_are_transposed
              ? GetTemporary(context, node, data->transposed_weights_index)
              : nullptr;

      TfLiteTensor* col2im =
          data->has_col2im ? GetTemporary(context, node, data->col2im_index)
                           : nullptr;

      TfLiteTensor* scratch_buffer;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, data->scratch_tensor_index,
                                    &scratch_buffer));

      GetInputSafe(context, inputs_[i][0], &output_shape_tensor);
      GetInputSafe(context, inputs_[i][2], &input);
      GetInputSafe(context, inputs_[i][1], &weights);
      GetInputSafe(context, inputs_[i][3], &bias);
      GetOutputSafe(context, outputs_[i][0], &output);

      // Resize any deferred dynamic tensors
      if (tflite::IsDynamicTensor(output)) {
        TF_LITE_ENSURE_OK(context,
                          ResizeTensor(context, output_shape_tensor, output));
      }
      if (data->has_col2im && tflite::IsDynamicTensor(col2im)) {
        TF_LITE_ENSURE_OK(
            context, ResizeCol2ImTensor(context, output_shape_tensor, weights,
                                        input, col2im));
      }

      if (tflite::IsDynamicTensor(scratch_buffer)) {
        TF_LITE_ENSURE_OK(context, ResizeTensor(context, output_shape_tensor,
                                                scratch_buffer));
      }

      // Get height and width of the output image.
      const int o_width = SizeOfDimension(output, 2);
      const int o_height = SizeOfDimension(output, 1);
      const int filter_w = SizeOfDimension(weights, 2);
      const int filter_h = SizeOfDimension(weights, 1);

      int unused_output_height, unused_output_width;
      data->padding = tflite::ComputePaddingHeightWidth(
          params->stride_height, params->stride_width, 1, 1, o_height, o_width,
          filter_h, filter_w, params->padding, &unused_output_height,
          &unused_output_width);

      tflite::ConvParams op_params;
      op_params.padding_type = PaddingType::kSame;
      op_params.padding_values.width = data->padding.width;
      op_params.padding_values.height = data->padding.height;
      op_params.padding_values.width_offset = data->padding.width_offset;
      op_params.padding_values.height_offset = data->padding.height_offset;
      op_params.stride_width = params->stride_width;
      op_params.stride_height = params->stride_height;
      // Need to flip the sign of input offset to add it directly to the
      // quantized buffer.
      op_params.input_offset = -input->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.quantized_activation_min = data->output_activation_min;
      op_params.quantized_activation_max = data->output_activation_max;

      ConvParams& cparams = op_params;
      const RuntimeShape& input_shape = GetTensorShape(input);
      const RuntimeShape& hwoi_ordered_filter_shape =
          GetTensorShape(transposed_weights);

      const RuntimeShape& output_shape = GetTensorShape(output);
      const RuntimeShape& col2im_shape = GetTensorShape(col2im);
      const RuntimeShape& scratch_shape = GetTensorShape(scratch_buffer);
      int32* output_multiplier = &crf[i][0];
      int32* output_shift = &crx[i][0];
      const int8_t* input_data = GetTensorData<int8>(input);
      const int8_t* hwoi_ordered_filter_data =
          GetTensorData<int8>(transposed_weights);

      int8_t* output_data = GetTensorData<int8>(output);
      int32_t* col2im_data = GetTensorData<int32>(col2im);
      int32_t* scratch_data = GetTensorData<int32>(scratch_buffer);
      CpuBackendContext* cpu_backend_context =
          CpuBackendContext::GetFromContext(context);

      const int batch_size =
          tflite::MatchingDim(input_shape, 0, output_shape, 0);
      const int input_image_size = input_shape.Dims(1) * input_shape.Dims(2);
      const int output_height = output_shape.Dims(1);
      const int output_width = output_shape.Dims(2);
      const int output_image_size = output_height * output_width;
      const int input_depth =
          tflite::MatchingDim(input_shape, 3, hwoi_ordered_filter_shape, 3);
      const int output_depth =
          tflite::MatchingDim(output_shape, 3, hwoi_ordered_filter_shape, 2);
      const int input_offset = input_image_size * input_depth;
      const int output_offset = output_image_size * output_depth;

      const int filter_height = hwoi_ordered_filter_shape.Dims(0);
      const int filter_width = hwoi_ordered_filter_shape.Dims(1);
      const int padding_top = cparams.padding_values.height;
      const int padding_bottom =
          cparams.padding_values.height + cparams.padding_values.height_offset;
      const int padding_left = cparams.padding_values.width;
      const int padding_right =
          cparams.padding_values.width + cparams.padding_values.width_offset;
      const int stride_height = cparams.stride_height;
      const int stride_width = cparams.stride_width;

      const int hwoi_ordered_filter_total_size =
          filter_height * filter_width * output_depth;

      const RuntimeShape& bias_shape = GetTensorShape(bias);
      const int32* bias_data = GetTensorData<int32>(bias);
      // if (has_bias) {
      //   bias_shape = GetTensorShape(bias);
      //   bias_data = GetTensorData<int32>(bias);
      // } else {
      //   TfLiteIntArray* dims = tensor->dims;
      //   const int dims_size = 1;
      //   const int32_t* dims_data = {output_depth};
      //   bias_shape = RuntimeShape(dims_size, dims_data);
      //   int32_t nbias_data[output_depth] = {0};
      //   bias_data = reinterpret_cast<const int32_t*>(&nbias_data);
      // }

      cpu_backend_gemm::MatrixParams<int8_t> lhs_params;
      lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
      lhs_params.rows = hwoi_ordered_filter_total_size;
      lhs_params.cols = input_depth;
      // Since our weight is symmetric quantized, the zp will always be 0.
      lhs_params.zero_point = 0;

      int32_t* scratch_data_p = scratch_data;
      std::fill_n(scratch_data, output_offset * batch_size,
                  static_cast<int32>(0));

      cpu_backend_gemm::MatrixParams<int8_t> rhs_params;
      rhs_params.order = cpu_backend_gemm::Order::kColMajor;
      rhs_params.rows = input_depth;
      rhs_params.cols = input_image_size;
      rhs_params.zero_point = -cparams.input_offset;

      cpu_backend_gemm::MatrixParams<int32_t> dst_params;
      dst_params.order = cpu_backend_gemm::Order::kColMajor;
      dst_params.rows = hwoi_ordered_filter_total_size;
      dst_params.cols = input_image_size;

      cpu_backend_gemm::GemmParams<int32_t, int32_t> gemm_params;

      const int gemm_input_rows = input_shape.Dims(3);
      const int gemm_input_cols = FlatSizeSkipDim(input_shape, 3);
      const int filter_rows = hwoi_ordered_filter_shape.Dims(0);
      const int filter_cols = FlatSizeSkipDim(hwoi_ordered_filter_shape, 0);
      const int output_rows = output_shape.Dims(3);
      const int output_cols =
          output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);

      //  Load & Reshape Input data to temporary buffers before offloading to
      //  DMA inbuffers
      int width = gemm_input_cols;
      int w = ((width + 3) - ((width + 3) % 4));
      int depth = gemm_input_rows;
      int d = ((depth + 15) - ((depth + 15) % 16));
      int d2 = depth * 2;
      int d3 = depth * 3;
      int d4 = depth * 4;

      int s_need = w * d / 4 + 1;
      int8_t inb0[s_need];
      int8_t inb1[s_need];
      int8_t inb2[s_need];
      int8_t inb3[s_need];
      int i_c = 0;
      int sums_curr = 0;

      int in_sum1[w / 4];
      int in_sum2[w / 4];
      int in_sum3[w / 4];
      int in_sum4[w / 4];

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
            unsigned char w0 = input_data[i0 + j];
            unsigned char w1 = input_data[i1 + j];
            unsigned char w2 = input_data[i2 + j];
            unsigned char w3 = input_data[i3 + j];
            ss0 += w0;
            ss1 += w1;
            ss2 += w2;
            ss3 += w3;
            inb0[i_c] = w0;
            inb1[i_c] = w1;
            inb2[i_c] = w2;
            inb3[i_c++] = w3;
          } else {
            inb0[i_c] = 0;
            inb1[i_c] = 0;
            inb2[i_c] = 0;
            inb3[i_c++] = 0;
          }
        }
        in_sum1[sums_curr] = (ss0);
        in_sum2[sums_curr] = (ss1);
        in_sum3[sums_curr] = (ss2);
        in_sum4[sums_curr++] = (ss3);
      }

      int* wb_0 = reinterpret_cast<int*>(&wb0[i][0]);
      int* wb_1 = reinterpret_cast<int*>(&wb1[i][0]);
      int* wb_2 = reinterpret_cast<int*>(&wb2[i][0]);
      int* wb_3 = reinterpret_cast<int*>(&wb3[i][0]);

      struct acc_container drv(acc, wb_0, wb_1, wb_2, wb_3, wt_sum1[i],
                               wt_sum2[i], wt_sum3[i], wt_sum4[i], crf[i],
                               crx_8[i]);
      drv.mdma = mdma;
      drv.profile = profile;
      drv.in_id = 0;
      int* inb_0 = reinterpret_cast<int*>(inb0);
      int* inb_1 = reinterpret_cast<int*>(inb1);
      int* inb_2 = reinterpret_cast<int*>(inb2);
      int* inb_3 = reinterpret_cast<int*>(inb3);
      drv.inb_0 = inb_0;
      drv.inb_1 = inb_1;
      drv.inb_2 = inb_2;
      drv.inb_3 = inb_3;
      drv.in_sum1 = in_sum1;
      drv.in_sum2 = in_sum2;
      drv.in_sum3 = in_sum3;
      drv.in_sum4 = in_sum4;

      int fake_bias[output_depth] = {};
      if (has_bias)
        drv.bias = biases[i];
      else
        drv.bias = fake_bias;

      // drv.bias = biases[i];
      drv.ra = cparams.output_offset;
      drv.rhs_offset = -rhs_params.zero_point;
      drv.lhs_offset = 0;
      drv.t.layer = dparams.layer;

      drv.output_depth = output_depth;
      drv.output_height = output_height;
      drv.output_width = output_width;
      drv.filter_height = filter_height;
      drv.filter_width = filter_width;
      drv.padding_top = padding_top;
      drv.padding_left = padding_left;
      drv.padding_bottom = padding_bottom;
      drv.padding_right = padding_right;
      drv.stride_height = stride_height;
      drv.stride_width = stride_width;

      drv.rows = rhs_params.cols;
      drv.cols = lhs_params.rows;
      drv.depth = rhs_params.rows;

      const int scratch_cols = scratch_shape.Dims(1) * scratch_shape.Dims(2);
      const int scratch_rows = scratch_shape.Dims(0) * scratch_shape.Dims(3);

      // LHS = Weights, RHS = Inputs, DST = GEMM Accumulated int32 data
      // Calls the gemm_driver to offload the CONV2D operation
      int8_params ts_lhs_params;
      int8_params ts_rhs_params;
      int32_params ts_dst_params;
      ts_lhs_params.Init(hwoi_ordered_filter_data, 0, lhs_params.rows,
                         lhs_params.cols, lhs_params.cols, 0);
      ts_rhs_params.Init(input_data, 1, rhs_params.rows, rhs_params.cols,
                         rhs_params.rows, 0);

      ts_dst_params.Init(col2im_data, 1, dst_params.rows, dst_params.cols, 0);

      cpu_backend_gemm::Gemm(lhs_params, hwoi_ordered_filter_data, rhs_params,
                             input_data, dst_params, col2im_data, gemm_params,
                             cpu_backend_context);

      // saveMatrixCSV(
      //     "aData/tconv/" + std::to_string(associated_nodes[i]) +
      //     "_del_wgt.csv", hwoi_ordered_filter_data, lhs_params.cols,
      //     lhs_params.rows);

      // saveMatrixCSV(
      //     "aData/tconv/" + std::to_string(associated_nodes[i]) +
      //     "_del_inp.csv", input_data, rhs_params.cols, rhs_params.rows);

      // saveMatrixCSV("aData/tconv/" + std::to_string(associated_nodes[i]) +
      //                   "_del_cpugemm_.csv",
      //               col2im_data, dst_params.cols, dst_params.rows);

      optimized_ops::Col2im(
          col2im_data, output_depth, output_height, output_width, filter_height,
          filter_width, padding_top, padding_left, padding_bottom,
          padding_right, stride_height, stride_width, scratch_data_p);

      // saveMatrixCSV("aData/tconv/" + std::to_string(associated_nodes[i]) +
      //                   "_del_out_col2im_cpu.csv",
      //               scratch_data_p, scratch_cols, scratch_rows);

      if (has_bias)
        optimized_ops::BiasAdd(scratch_data_p, bias_data, batch_size,
                               output_height, output_width, output_depth);

      const int32_t output_min = std::numeric_limits<int8_t>::min();
      const int32_t output_max = std::numeric_limits<int8_t>::max();
      optimized_ops::Quantize(output_multiplier, output_shift, output_depth,
                              output_shape.FlatSize(), cparams.output_offset,
                              output_min, output_max, scratch_data,
                              output_data);

      saveMatrixCSV("aData/tconv/" + std::to_string(associated_nodes[i]) +
                        "_del_out_cpu.csv",
                    output_data, scratch_cols, scratch_rows);

      tflite_vm_tconv_sim::Entry(drv, col2im_data, output_data);

      // saveMatrixCSV("aData/tconv/" + std::to_string(associated_nodes[i]) +
      //                   "_del_accgemm.csv",
      //               col2im_data, dst_params.cols, dst_params.rows);

      saveMatrixCSV("aData/tconv/" + std::to_string(associated_nodes[i]) +
                        "_del_out_acc.csv",
                    output_data, scratch_cols, scratch_rows);

      dparams.layer++;
    }
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

  std::vector<int*> biases;
  std::vector<OpData*> opdatas;
  std::vector<std::vector<int>> crf;
  std::vector<std::vector<int>> crx;
  std::vector<std::vector<int8_t>> crx_8;

  std::vector<TfLiteTransposeConvParams*> tparams;

 private:
  const TCONVSimDelegateOptions options_;
};

// TCONVSimDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class TCONVSimDelegate : public SimpleDelegateInterface {
 public:
  explicit TCONVSimDelegate(const TCONVSimDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports TCONV op
    if (kTfLiteBuiltinTransposeConv != registration->builtin_code) return false;

    // This delegate requires at least 3 inputs.
    if (node->inputs->size < 3) return false;

    // This delegate only supports int8 types.
    for (int i = 1; i < 3; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    // Ensures output shape tensor is supports int32 type
    auto& tensor = context->tensors[node->inputs->data[0]];
    if (tensor.type != kTfLiteInt32) return false;

    if (node->inputs->size == 4) {
      // Ensures bias tensor is supports int32 type
      auto& tensor2 = context->tensors[node->inputs->data[3]];
      if (tensor2.type != kTfLiteInt32) return false;
    }

    // Adds node for delegation
    dparams.delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "TCONVSimDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<TCONVSimDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const TCONVSimDelegateOptions options_;
};

}  // namespace tconvsim_test
}  // namespace tflite

TCONVSimDelegateOptions TfLiteTCONVSimDelegateOptionsDefault() {
  TCONVSimDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this tconvsim test delegate
  // will not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteTCONVSimDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteTCONVSimDelegateCreate(
    const TCONVSimDelegateOptions* options) {
  std::unique_ptr<tflite::tconvsim_test::TCONVSimDelegate> tconvsim(
      new tflite::tconvsim_test::TCONVSimDelegate(
          options ? *options : TfLiteTCONVSimDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(tconvsim), kTfLiteDelegateFlagsAllowDynamicTensors);
}

// Destroys a delegate created with `TfLiteTCONVSimDelegateCreate` call.
void TfLiteTCONVSimDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
