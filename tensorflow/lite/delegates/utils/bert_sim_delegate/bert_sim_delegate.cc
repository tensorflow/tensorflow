
#include "tensorflow/lite/delegates/utils/bert_sim_delegate/bert_sim_delegate.h"
#include <fstream>
#include <iostream>
#include <utility>

#include "accelerator/driver/fc_driver.h"
#include "tensorflow/lite/delegates/utils/bert_sim_delegate/util.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_integrator/systemc_integrate.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

// Some variables needs to be defined across multiple instances of the delegate
ACCNAME* acc;
struct del_params dparams;
struct sysC_sigs* scs;
struct Profile* profile;

namespace tflite {
namespace bert_sim_test {

// BertSim delegate kernel.
class BertSimDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit BertSimDelegateKernel(const BertSimDelegateOptions& options)
      : options_(options) {}

  // Runs once per delegate partition
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Init SystemC Modules & Profilier
    if (!dparams.init) {
      static struct sysC_sigs _scs(1);
      static ACCNAME accelerator("accelerator");
      static struct Profile _profile;
      sysC_init();
      systemC_binder(&accelerator, &_scs);
      acc = &accelerator;
      scs = &_scs;
      profile = &_profile;
      dparams.init = true;

      std::cout << "===========================" << std::endl;
      std::cout << "Initialised the SystemC Modules" << std::endl;
      std::cout << "FC-GEMM Accelerator";
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
    wgt_sum.resize(params->nodes_to_replace->size);
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
      TfLiteFullyConnectedParams* cparam =
          reinterpret_cast<TfLiteFullyConnectedParams*>(
              delegated_node->builtin_data);
      OpData* opdata = reinterpret_cast<OpData*>(delegated_node->user_data);

      cparams[i] = cparam;
      opdatas[i] = opdata;
    }
    return kTfLiteOk;
  }

  // Runs once per node before inference/invoke()
  // This function preloads weights, allocates additional tensors, calculates
  // quantization parameters For more info look into
  // "tensorflow/lite/kernels/fullyconnected.cc" for the default implementation
  // for FullyConnected Nodes
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    int node_count = inputs_.size();
    int out_tid = 0;
    for (int i = 0; i < node_count; i++) {
      TfLiteFullyConnectedParams* params = cparams[i];
      OpData* data = opdatas[i];

      TfLiteTensor* output;
      const TfLiteTensor* input;
      const TfLiteTensor* filter;
      const TfLiteTensor* bias;

      GetOutputSafe(context, outputs_[i][0], &output);
      GetInputSafe(context, inputs_[i][0], &input);
      GetInputSafe(context, inputs_[i][1], &filter);
      if (inputs_[i].size() == 3 && inputs_[i][2] >= 0) {
        GetInputSafe(context, inputs_[i][2], &bias);
        biases[i] = bias->data.i32;
      } else {
        biases[i] = nullptr;
        bias = nullptr;
      }

      // Get Qaunt Params.
      double real_multiplier = 0.0;
      int exponent;
      GetQuantizedConvolutionMultipler(context, input, filter, bias, output,
                                       &real_multiplier);
      QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
      CalculateActivationRangeQuantized(context, params->activation, output,
                                        &data->output_activation_min,
                                        &data->output_activation_max);

      // Resize output.
      int input_size = 1;
      for (int i = 0; i < input->dims->size; i++)
        input_size *= input->dims->data[i];
      const int batch_size = input_size / filter->dims->data[1];
      const int num_units = filter->dims->data[0];
      const int out_dim1 = batch_size;
      const int out_dim2 = num_units;
      TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
      output_size->data[0] = out_dim1;
      output_size->data[1] = out_dim2;
      auto output_status = context->ResizeTensor(context, output, output_size);
      if (output_status != kTfLiteOk) return output_status;

      int temp_out_id;
      bool req_temp_out = outputs_[i][0] != node->outputs->data[out_tid];
      if (!req_temp_out) out_tid++;

      TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(
          context, node, req_temp_out, outputs_[i][0], temp_out_id,
          inputs_[i][0], inputs_[i][1]));

      int k = node->outputs->data[out_tid];

      if (req_temp_out) {
        node->temporaries->data[temp_out_id] = outputs_[i][0];
        TfLiteIntArray* temp_out_tensor_size = TfLiteIntArrayCreate(2);
        temp_out_tensor_size->data[0] = output_size->data[0];
        temp_out_tensor_size->data[1] = output_size->data[1];

        TfLiteTensor* temp_out_tensor = &context->tensors[outputs_[i][0]];
        temp_out_tensor->type = kTfLiteInt8;
        temp_out_tensor->allocation_type = kTfLiteArenaRw;
        auto temp_out_tensor_status = context->ResizeTensor(
            context, temp_out_tensor, temp_out_tensor_size);
        if (temp_out_tensor_status != kTfLiteOk) return temp_out_tensor_status;
      }
    }
    return kTfLiteOk;
  }

  // Runs once per node during inference/invoke()
  // This function executes the operations required by node by offloading the
  // computation to the fc_driver For more info look into
  // "tensorflow/lite/kernels/fullyconnected.cc" for the default implementation
  // for FullyConnected Nodes
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

      const TfLiteTensor* bias;
      bool isBias = biases[i] ? true : false;
      if (isBias)
        GetInputSafe(context, inputs_[i][2], &bias);
      else
        bias = nullptr;

      const int8* input_data = input->data.int8;
      const int8* filter_data = filter->data.int8;
      int8* output_data = output->data.int8;
      const int32_t* bias_data =
          (bias != nullptr ? reinterpret_cast<int32_t*>(bias->data.raw)
                           : nullptr);

      FullyConnectedParams op_params;
      op_params.input_offset = -input->params.zero_point;
      op_params.weights_offset = -filter->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.output_multiplier = data->output_multiplier;
      op_params.output_shift = data->output_shift;
      op_params.quantized_activation_min = data->output_activation_min;
      op_params.quantized_activation_max = data->output_activation_max;
      op_params.lhs_cacheable = IsConstantTensor(filter);
      op_params.rhs_cacheable = IsConstantTensor(input);
      const int32_t output_offset = op_params.output_offset;
      const int32_t lhs_offset = -op_params.weights_offset;
      const int32_t rhs_offset = -op_params.input_offset;
      const int32_t output_multiplier = op_params.output_multiplier;
      const int output_shift = op_params.output_shift;
      const int32_t output_activation_min = op_params.quantized_activation_min;
      const int32_t output_activation_max = op_params.quantized_activation_max;
      RuntimeShape input_shape =
          RuntimeShape(input->dims->size, input->dims->data);
      RuntimeShape filter_shape =
          RuntimeShape(filter->dims->size, filter->dims->data);
      RuntimeShape output_shape =
          RuntimeShape(output->dims->size, output->dims->data);
      const int output_dim_count = output_shape.DimensionsCount();
      const int filter_dim_count = filter_shape.DimensionsCount();
      const int output_depth = output_shape.Dims(1);
      const int filter_rows = filter_shape.Dims(filter_dim_count - 2);
      const int filter_cols = filter_shape.Dims(filter_dim_count - 1);
      const int batches = output_shape.Dims(0);
      const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

      int N = batches;
      int M = output_depth;
      int K = accum_depth;
      int rfactor = 16;
      int pN = roundUp(N, rfactor);
      int pM = roundUp(M, rfactor);
      int pK = roundUp(K, rfactor);

      std::vector<int> in_sum;
      std::vector<int> wt_sum;
      int* idims = input->dims->data;
      int* wdims = filter->dims->data;
      int8_t* padded_input = new int8_t[pN * pK];
      int8_t* padded_weights = new int8_t[pM * pK];
      int8_t* padded_output = new int8_t[pM * pN];

      // Calls the fc_driver to re-shape TFLite input/weight tensor and also
      // produces vector of sums from the tensor's rows (required for
      // re-quantization)
      precal_sum_load_pad(input->data.int8, N, K, padded_input, in_sum);
      precal_sum_load_pad(filter->data.int8, M, K, padded_weights, wt_sum);

      // acc_container is used to wrap all the paramters the
      // fc_driver/accelerator needs from the delegate
      struct acc_container drv;
      drv.scs = scs;
      drv.profile = profile;
      drv.acc = acc;
      drv.layer = associated_nodes[i];
      drv.pN = pN;
      drv.pM = pM;
      drv.pK = pK;
      drv.N = N;
      drv.M = M;
      drv.K = K;
      drv.padded_input = padded_input;
      drv.padded_weights = padded_weights;
      drv.padded_output = padded_output;
      drv.in_sum = &in_sum[0];
      drv.wt_sum = &wt_sum[0];
      drv.crx = output_shift;
      drv.crf = output_multiplier;
      drv.ra = output_offset;
      drv.rhs_offset = -rhs_offset;
      drv.lhs_offset = -lhs_offset;
      if (!isBias)
        drv.bias = new int32_t[pM]();
      else
        drv.bias = biases[i];

      // Calls the fc_driver to offload the FC operation
      drv.start_count = dparams.start_count;
      tflite_bertsim::Entry(drv);
      dparams.start_count = drv.start_count;

      // Calls the fc_driver unpack/unpad result to TFLite tensor
      store_unpad(padded_output, N, M, output_data);
      if (!isBias) delete[] drv.bias;

      dparams.layer++;
      dparams.delegated_nodes--;
    }

    // Saves profilier records once all delegated nodes are executed
    if (dparams.delegated_nodes == 0) {
      profile->saveCSVRecords("bert_sim");
    }
    return kTfLiteOk;
  }

  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_, associated_nodes;

  std::vector<std::vector<int>> wgt_sum;
  std::vector<int> weight_offsets;

  std::vector<uint32_t*> del_weights;

  std::vector<int*> biases;
  std::vector<int*> crf;
  std::vector<int8_t*> crx;

  std::vector<OpData*> opdatas;
  std::vector<TfLiteFullyConnectedParams*> cparams;

 private:
  const BertSimDelegateOptions options_;
};

// BertSimDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class BertSimDelegate : public SimpleDelegateInterface {
 public:
  explicit BertSimDelegate(const BertSimDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // Only supports FC ops
    if (kTfLiteBuiltinFullyConnected != registration->builtin_code)
      return false;

    if (node->inputs->size != 3 && node->inputs->size != 2) return false;
    // This delegate only supports int8 types.
    for (int i = 0; i < 2; ++i) {
      auto& tensor = context->tensors[node->inputs->data[i]];
      if (tensor.type != kTfLiteInt8) return false;
    }

    if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
      auto& tensor = context->tensors[node->inputs->data[2]];
      if (tensor.type != kTfLiteInt32 && tensor.type <= 16) return false;
    }

    // FC
    dparams.delegated_nodes++;
    // cout << dparams.delegated_nodes << endl;
    return true;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "BertSimDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<BertSimDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const BertSimDelegateOptions options_;
};

}  // namespace bert_sim_test
}  // namespace tflite

BertSimDelegateOptions TfLiteBertSimDelegateOptionsDefault() {
  BertSimDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this bert_sim test delegate
  // will not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteBertSimDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteBertSimDelegateCreate(
    const BertSimDelegateOptions* options) {
  std::unique_ptr<tflite::bert_sim_test::BertSimDelegate> bert_sim(
      new tflite::bert_sim_test::BertSimDelegate(
          options ? *options : TfLiteBertSimDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(bert_sim));
}

// Destroys a delegate created with `TfLiteBertSimDelegateCreate` call.
void TfLiteBertSimDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
