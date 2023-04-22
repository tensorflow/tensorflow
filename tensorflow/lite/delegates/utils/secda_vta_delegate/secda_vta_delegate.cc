/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/utils/rvta_delegate/rvta_delegate.h"

#include <utility>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

#include "tensorflow/lite/delegates/utils/rvta_delegate/util.h"

#include "tensorflow/lite/delegates/utils/rvta_delegate/driver/axi_mm_api.h"
#include "tensorflow/lite/delegates/utils/rvta_delegate/driver/conv2d_driver.h"
#include "tensorflow/lite/delegates/utils/rvta_delegate/driver/mm_helper.h"
#include "tensorflow/lite/delegates/utils/rvta_delegate/driver/vta_gemm.h"

bool acc_init = false;
int layer = 0;
int cweight_offset = 0;

unsigned long long* opc_mem;
unsigned int* uop_mem;

unsigned long long* inp_mem;
unsigned long long* wgt_mem;
unsigned long long* bias_mem;
unsigned long long* crf_mem;
unsigned long long* crx_mem;
unsigned long long* out_mem;

int* acc;
unsigned int vta_count = 0;
struct times vta_t;
int delegated_nodes = 0;

int ins_count = 0;

namespace tflite {
namespace rvta_test {

// RVta delegate kernel.
class RVtaDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit RVtaDelegateKernel(const RVtaDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Init SystemC

    if (!acc_init) {
      acc = getArray(acc_address, PAGE_SIZE);
      opc_mem = mm_init_writable<unsigned long long>(opc_addr, MM_BL);
      uop_mem = mm_init_writable<unsigned int>(uop_addr, MM_BL);
      inp_mem = mm_init_writable<unsigned long long>(in_addr, MM_BL);
      // wgt_mem = mm_init_writable<unsigned long long>(wgt_addr, MM_BL);
      wgt_mem = mm_init_writable<unsigned long long>(wgt_addr, MM_BL*4);
      // bias_mem = mm_init_writable<unsigned long long>(bias_addr, MM_BL);
      bias_mem = mm_init_writable<unsigned long long>(bias_addr, MM_BL * 4);
      crf_mem = mm_init_writable<unsigned long long>(crf_addr, MM_BL);
      crx_mem = mm_init_writable<unsigned long long>(crx_addr, MM_BL);
      // out_mem = mm_init<unsigned long long>(out_addr, MM_BL);
      out_mem = mm_init<unsigned long long>(out_addr, MM_BL * 4);

      writeMappedReg(acc, 0x14, 0);
      writeMappedReg(acc, 0x24, 1);
      writeMappedReg(acc, 0x34, opc_addr / 8);
      writeMappedReg(acc, 0x3c, uop_addr / 4);
      writeMappedReg(acc, 0x44, in_addr / 8);
      writeMappedReg(acc, 0x4c, wgt_addr / 8);
      writeMappedReg(acc, 0x54, bias_addr / 8);
      writeMappedReg(acc, 0x5c, out_addr / 8);
      writeMappedReg(acc, 0x64, crf_addr / 8);
      writeMappedReg(acc, 0x6c, crx_addr / 8);

      acc_init = true;
      std::cout << "===========================" << std::endl;
      std::cout << "Memory Mapped Buffers" << std::endl;
      std::cout << "VTA";
#ifdef ACC_NEON
      std::cout << " with Neon";
#endif
      std::cout << std::endl;
      std::cout << "===========================" << std::endl;
      writeMappedReg(acc, 0x24, 0);
    }

    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);

    opdatas.resize(params->nodes_to_replace->size);
    cparams.resize(params->nodes_to_replace->size);

    wt_sum.resize(params->nodes_to_replace->size);
    biases.resize(params->nodes_to_replace->size);
    crf.resize(params->nodes_to_replace->size);
    crx.resize(params->nodes_to_replace->size);
    w_offsets.resize(params->nodes_to_replace->size);

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

      // data->per_channel_output_multiplier.resize(channels_out);
      data->per_channel_output_shift.resize(channels_out);
      crf[i].resize(channels_out);
      crx[i].resize(channels_out);

      // cout << "channels_out: " << channels_out << endl;
      // crf[i] = new int[channels_out];
      // crx[i] = new int8_t[channels_out];

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

      // // Might need to transform this in a bias matrix
      // biases[i] = bias->data.i32;
      // int* dims = filter->dims->data;
      // const int8* filter_data = filter->data.int8;
      // int M = dims[0];
      // int K = dims[1] * dims[2] * dims[3];
      // int N = out_height * out_width * batches;
      // int pN = roundup(N, 16);
      // int pM = roundup(M, 16);
      // int pK = roundup(K, 16);

      // unsigned int* wgt_mm = (unsigned int*)(wgt_mem) + (cweight_offset / 4);
      // w_offsets[i] = cweight_offset;
      // int8_t* padded_weights = (int8_t*)&wgt_mm[0];
      // precal_sum_load_pad(filter->data.int8, M, K, padded_weights,
      // wt_sum[i]); cweight_offset += pM * pK;

      // Might need to transform this in a bias matrix
      biases[i] = bias->data.i32;
      int* dims = filter->dims->data;
      preload_weights<int>(filter->data.int8, dims, wt_sum[i]);

      const int8* filter_data = filter->data.int8;
      unsigned int* wgt_mm = (unsigned int*)(wgt_mem) + (cweight_offset / 4);
      w_offsets[i] = cweight_offset;

      int M = dims[0];
      int K = dims[1] * dims[2] * dims[3];
      int N = out_height * out_width * batches;
      int pN = roundup(N, 16);
      int pM = roundup(M, 16);
      int pK = roundup(K, 16);

      int8_t** padded_weights = new int8_t*[pM]();
      for (int j = 0; j < pM; ++j) padded_weights[j] = new int8_t[pK]();

      pad_matrix(M, K, 16, 16, filter_data, padded_weights);
      tflite_secda::packBufferBlock<uint32_t, 32, int8_t, 8>(
          wgt_mm, padded_weights, 0, pM, VTA_BLOCK_OUT, 0, pK, VTA_BLOCK_IN);
      cweight_offset += pM * pK;

      for (int j = 0; j < pM; ++j) delete[] padded_weights[j];
      delete[] padded_weights;
    }
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    prf_start(5);
    int node_count = inputs_.size();
    for (int i = 0; i < node_count; i++) {
      prf_start(0);
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
      op_params.weights_offset = -filter->params.zero_point;
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

      const int32_t lhs_offset = -op_params.weights_offset;
      const int32_t rhs_offset = -op_params.input_offset;

      int N = gemm_input_cols;
      int M = output_rows;
      int K = gemm_input_rows;

      int rfactor = 16;
      int pN = roundup(N, rfactor);
      int pM = roundup(M, rfactor);
      int pK = roundup(K, rfactor);

      int8_t* padded_input = (int8_t*)&inp_mem[0];
      int8_t* padded_output = (int8_t*)&out_mem[0];
      vector<int> in_sum;
      in_sum.resize(pN);

      prf_start(1);
      precal_sum_load_pad(const_cast<int8_t*>(gemm_input_data), N, K,
                          padded_input, &in_sum[0]);
      prf_end(1, vta_t.ipack);
      int preload_weight_address = wgt_addr + w_offsets[i];
      writeMappedReg(acc, 0x4c, preload_weight_address / 8);

      struct conv2d_driver drv;
      drv.bias = biases[i];
      // drv.in_sum = in_sum;
      drv.in_sum = &in_sum[0];
      drv.wt_sum = &wt_sum[i][0];
      drv.crf = &crf[i][0];
      drv.crx = &crx[i][0];
      drv.ra = output_offset;
      // drv.rhs_offset = input_offset;
      // drv.lhs_offset = 0;
      drv.rhs_offset = -rhs_offset;
      drv.lhs_offset = -lhs_offset;
      drv.pN = pN;
      drv.pM = pM;
      drv.pK = pK;
      drv.N = N;
      drv.M = M;
      drv.K = K;

      drv.acc = acc;
      drv.opc_mm = opc_mem;
      drv.uop_mm = uop_mem;
      drv.bias_mm = bias_mem;
      drv.crf_mm = crf_mem;
      drv.crx_mm = crx_mem;
      drv.layer = layer;

      drv.t = vta_t;
      drv.vta_count = vta_count;
      drv.ins_count = ins_count;
      tflite_secda::Entry<int>(drv);
      vta_count = drv.vta_count;
      ins_count = drv.ins_count;
      vta_t = drv.t;

      prf_start(2);
      store_unpad(padded_output, N, M, output_data);
      prf_end(2, vta_t.unpack);

      // std::string mname = "conv_v1";
      // std::string mw = "";
      // std::string mi = "";
      // std::string mo = "";

      // int8_t* packed_weights = (int8_t*)(wgt_mem);

      // {
      //   ofstream myfile;
      //   myfile.open("a_Vta/" + mname + mw + "/del_wgt_" +
      //               std::to_string(associated_nodes[i]) + ".csv");
      //   const int8_t* res_pointer = packed_weights;
      //   // const int8_t* res_pointer = filter_data;
      //   int index = 0;
      //   for (int c = 0; c < M; c++) {
      //     myfile << endl;
      //     for (int r = 0; r < K; r++) {
      //       myfile << (int)res_pointer[index] << ",";
      //       index++;
      //     }
      //   }
      //   myfile.close();
      // }

      // {
      //   ofstream myfile;
      //   myfile.open("a_Vta/" + mname + mi + "/del_in_" +
      //               std::to_string(associated_nodes[i]) + ".csv");
      //   // const int8_t* res_pointer = gemm_input_data;
      //   const int8_t* res_pointer = padded_input;
      //   int index = 0;
      //   for (int c = 0; c < N; c++) {
      //     myfile << endl;
      //     for (int r = 0; r < K; r++) {
      //       myfile << (int)res_pointer[index] << ",";
      //       index++;
      //     }
      //   }
      //   myfile.close();
      // }

      // {
      //   ofstream myfile;
      //   myfile.open("a_Vta/" + mname + mo + "/del_out_" +
      //               std::to_string(associated_nodes[i]) + ".csv");
      //   int8_t* res_pointer = padded_output;
      //   // int8_t* res_pointer = output_data;
      //   int index = 0;
      //   for (int c = 0; c < N; c++) {
      //     myfile << endl;
      //     for (int r = 0; r < M; r++) {
      //       myfile << (int)res_pointer[index] << ",";
      //       index++;
      //     }
      //   }
      //   myfile.close();
      // }

      layer++;
      delegated_nodes--;
    }

    prf_end(5, vta_t.conv_total);
    if (delegated_nodes == 0) {
      vta_t.print();
      cout << "ins_count:" << ins_count << endl;
    }
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;

  std::vector<std::vector<int>> wt_sum;
  std::vector<int*> biases;
  // std::vector<int*> crf;
  // std::vector<int8_t*> crx;

  std::vector<std::vector<int>> crf;
  std::vector<std::vector<int8_t>> crx;


  std::vector<int> w_offsets;

  std::vector<OpData*> opdatas;
  std::vector<TfLiteConvParams*> cparams;

 private:
  const RVtaDelegateOptions options_;
};

// RVtaDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class RVtaDelegate : public SimpleDelegateInterface {
 public:
  explicit RVtaDelegate(const RVtaDelegateOptions& options)
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
      // for (int j = 0; j < tensor.dims->size; j++)
      //   cout << (int)tensor.dims->data[j] << " , ";
      // cout << endl;
      if (tensor.dims->data[0] == 1 && tensor.dims->data[1] == 1 &&
          tensor.dims->data[2] == 1) {
        // cout << "Skipping This Layer" << endl;
        return false;
      }
      if (tensor.type != kTfLiteInt8) {
        // cout << "Skipping This Layer" << endl;
        return false;
      }
    }
    // cout << "------------------------" << endl;
    auto& tensor = context->tensors[node->inputs->data[2]];
    if (tensor.type != kTfLiteInt32) return false;

    // CONV2D
    delegated_nodes++;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "RVtaDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<RVtaDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const RVtaDelegateOptions options_;
};

}  // namespace rvta_test
}  // namespace tflite

RVtaDelegateOptions TfLiteRVtaDelegateOptionsDefault() {
  RVtaDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this rvta test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteRVtaDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteRVtaDelegateCreate(const RVtaDelegateOptions* options) {
  std::unique_ptr<tflite::rvta_test::RVtaDelegate> rvta(
      new tflite::rvta_test::RVtaDelegate(
          options ? *options : TfLiteRVtaDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(rvta));
}

// Destroys a delegate created with `TfLiteRVtaDelegateCreate` call.
void TfLiteRVtaDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
