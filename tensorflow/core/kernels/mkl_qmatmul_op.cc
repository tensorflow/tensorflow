/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Implements a quantized eight-bit version of the matmul operation with bias,
// relu and requantization fusion support utilizing mkldnn u8s8s32 inner
// product API. Right now, this version can support
//   - Input: quantized as uint8 via either MIN_FIRST or SCALE mode.
//            SCALE mode is selected when input is guaranteed to be non-
//            negative, e.g., MatMul is fed by Relu. Otherwise, MIN_FIRST is
//            selected.
//   - Weight: quantized to int8 via SCALE mode.
//   - Bias: float32/int32. For int32, it is quantized according to input and
//           filter min-max values.
// Other than that, this op does not support other input combination yet.
// When input is quantized to uint8 via MIN_FIRST, bias needs compensation.
// The detailed algorithm is illustrated as below:
//
// Af32 is the original fp32 activation 2D tensor.
// Min(Af32) is the minimum scalar value of Af32.
// Max(Af32) is the maximum scalar value of Af32.
// Qa is the quantization scale for activation.
// Au8 is the quantized unsigned int8 activation tensor.
// With SCALE quantization (used for non-negative Af32), Qa and Au8 can be
// calculated as below:
//    Qa = 255.0 / Max(Af32)
//    Au8 = round(Qa * Af32).
// With MIN_FIRST quantization, Q'a and A'u8 can be calculated as below:
//    Q'a = 255.0 / (Max(Af32) - Min(Af32))
//    A'u8 = round(Q'a * (Af32 - Min(Af32) * ones(Af32))),
// where, ones(.) is a tensor of all 1s with the same shape of its argument and
// round(.) rounds a number to its nearest integer.
//
// Wf32 is the original fp32 2D weight tensor.
// MaxAbs(Wf32) is the maximum absolute scalar value of Wf32.
// Qw is the quantization scale of weight.
// Ws8 is the quantized signed int8 weight tensor.
// Qw and Ws8 can be calculated as below:
//    Qw = 127.0 / MaxAbs(Wf32)
//    Ws8 = round(Qw * Wf32).
//
// Bf32 is the original fp32 1D bias tensor matching the innermost dim of
// Wf32.
// With SCALE quantization of activation, the scaled bias, Bs32, is calculated
// as below:
//      Bs32 = Qa * Qw * Bf32.
// With MIN_FIRST quantization of activation, the scaled bias tensor with
// compensation, B's32, is calculated as below:
//      B's32 = Q'a * Qw * Bf32 + Q'a * Qw * Min(Af32) * 1 * Wf32
//            = Q'a * Qw * Bf32 + Q'a * Min(Af32) * 1 * Ws8.
// where, 1 denotes a row vector matching the outermost dim of Wf32.
//
// The QuantizedMatMulWithBias op calculates 32bit integer output as below:
//  - with SCALE activation quantization:
//    Xs32 = Au8 * Ws8 + 1' * Bs32
//         = Qa * Qw * Af32 * Wf32  + Qa * Qw * 1' * Bf32
//         = Qa * Qw * (Af32 * Wf32 + 1' * Bf32) = Qa * Qw * Xf32,
//    where, 1' denotes a column vector matching the outermost dim of Af32 and
//    Xf32 represents the output of original fp32 MatMul with BiasAdd fusion.
//
//  - with MIN_FIRST activation quantization:
//    Xs32 = A'u8 * Ws8 + 1' * B's32
//         = Q'a * (Af32 - Min(Af32) * ones(Af32)) * Qw * Wf32 +
//           Q'a * Qw * 1' * Bf32 + Q'a * Qw * Min(Af32) * 1' * 1 * Wf32
//         = Q'a * Qw * (Af32 * Wf32 + 1' * Bf32)
//         = Q'a * Qw * Xf32.
//    Note that 1' * 1 = ones(Af32).
//
// The QuantizedMatMulWithBiasAndRelu op does the same calculation as above
// except adding relu function for the 32bit integer output.
//
// The QuantizedMatMulWithBiasAndReluAndRequantize op does one more step of
// requantize calculation based on above. Since the fusion ends with a Relu the
// activation Xf32 at Relu, in the original fp32 graph, is guaranteed to be
// non-negative. The requantize scale Qr is calculated from offline calibration.
//    Qr = 255 / Max(Xf32)
//    Xu8 = Qr * Xf32.
//
// More information of this implementation can be found in
// https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training
#ifdef INTEL_MKL

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl_matmul_ops_common.h"
#include "tensorflow/core/kernels/mkl_quantized_conv_ops.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"

namespace {
enum {
  QUANTIZE_MODE_MIN_FIRST,
  QUANTIZE_MODE_SCALED,
};
}  // namespace

namespace tensorflow {

template <typename Device, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnQuantizedMatMulOp : public MklDnnMatMulOpBase<Tweight, Toutput> {
 public:
  virtual ~MklDnnQuantizedMatMulOp() {
    if (this->input_bias_ != nullptr) {
      delete this->input_bias_;
      input_bias_ = nullptr;
    }
    if (this->scaled_bias_ != nullptr) {
      delete this->scaled_bias_;
      scaled_bias_ = nullptr;
    }
    if (this->comp_bias_ != nullptr) {
      delete this->comp_bias_;
      comp_bias_ = nullptr;
    }
  }

  float* GetCompBiasBuffer(int size) {
    if (comp_bias_ == nullptr) {
      comp_bias_ = new float[size];
    }
    return comp_bias_;
  }

  explicit MklDnnQuantizedMatMulOp(OpKernelConstruction* context)
      : MklDnnMatMulOpBase<Tweight, Toutput>(context) {
    string mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("input_quant_mode", &mode_string));
    if (mode_string == "MIN_FIRST") {
      mode_ = QUANTIZE_MODE_MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QUANTIZE_MODE_SCALED;
    } else {
      context->CtxFailure(errors::InvalidArgument(
          "Quantization mode must be either MIN_FIRST or SCALED, but received ",
          mode_string));
    }
    this->is_weight_const_ = false;
    if (context->HasAttr("is_weight_const")) {
      OP_REQUIRES_OK(context, context->GetAttr("is_weight_const",
                                               &(this->is_weight_const_)));
    }
  }

  void Compute(OpKernelContext* context) override {
    try {
      // Input tensors
      const Tensor& src_tensor = MklGetInput(context, this->kInputIndexSrc);
      const Tensor& weight_tensor =
          MklGetInput(context, this->kInputIndexWeight);
      const Tensor& bias_tensor = MklGetInput(context, this->kInputIndexBias);

      MklDnnShape src_mkl_shape, weight_mkl_shape;
      GetMklShape(context, this->kInputIndexSrc, &src_mkl_shape);
      GetMklShape(context, this->kInputIndexWeight, &weight_mkl_shape);
      OP_REQUIRES(context, !weight_mkl_shape.IsMklTensor(),
                  errors::InvalidArgument("Weight should not be in "
                                          "MKL Layout"));

      MklDnnData<Tinput> src(&(this->cpu_engine_));
      MklDnnData<Tweight> weight(&(this->cpu_engine_));

      memory::dims src_dims, weight_dims;
      memory::dims dst_dims_tf_order, dst_dims_mkl_order;

      // Get shapes of input tensors in MKL-DNN order
      auto src_tf_shape = src_mkl_shape.IsMklTensor()
                              ? src_mkl_shape.GetTfShape()
                              : src_tensor.shape();
      auto weight_tf_shape = weight_mkl_shape.IsMklTensor()
                                 ? weight_mkl_shape.GetTfShape()
                                 : weight_tensor.shape();

      src_dims = TFShapeToMklDnnDims(src_tf_shape);
      weight_dims = TFShapeToMklDnnDims(weight_tf_shape);
      dst_dims_mkl_order = {static_cast<int>(src_tf_shape.dim_size(0)),
                            static_cast<int>(weight_tf_shape.dim_size(1))};

      // Weight dims need to be reversed to create inner-product forward
      // descriptor
      weight_dims = {static_cast<int>(weight_tf_shape.dim_size(1)),
                     static_cast<int>(weight_tf_shape.dim_size(0))};

      // Create memory for user data.
      // Describe how the inputs and outputs of inner-product look like. Also
      // specify buffers containing actual input and output data.
      Tensor* dst_tensor = nullptr;
      auto input_output_fmt = MEMORY_FORMAT::nc;
      auto input_output_fmt_mkldnn = MKL_TENSOR_FORMAT_NC;

      // If input is in MKL layout, then simply take input layout; otherwise,
      // construct input TF layout. For TF layout, although input shape
      // (src_dims) required is in MKL-DNN order, the layout is Tensorflow's
      // layout depending on data format.
      auto src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<Tinput>(), input_output_fmt);
      src.SetUsrMem(src_md, &src_tensor);

      // Although weight shape (weight_dims) required is in MKL-DNN order,
      // the layout is TensorFlow's layout.
      auto weight_md = weight_mkl_shape.IsMklTensor()
                           ? weight_mkl_shape.GetMklLayout()
                           : memory::desc(weight_dims, MklDnnType<Tweight>(),
                                          MEMORY_FORMAT::io);
      weight.SetUsrMem(weight_md, &weight_tensor);

      MklDnnMatMulFwdPrimitive<float, Tinput, Tweight, Tbias, Toutput>*
          matmul_fwd = nullptr;
      memory::dims bias_dims = {static_cast<int>(bias_tensor.dim_size(0))};

      MklDnnMatMulFwdParams matmul_fwd_dims(src_dims, weight_dims, bias_dims,
                                            dst_dims_mkl_order);

      // Extend the basic parameters for data types and fusions.
      this->ExtendMklDnnMatMulFwdParams(context, matmul_fwd_dims);

      // Get a MatMul fwd from primitive pool.
      matmul_fwd =
          MklDnnMatMulFwdPrimitiveFactory<float, Tinput, Tweight, Tbias,
                                          Toutput>::Get(matmul_fwd_dims, 0);

      // Allocate output Tensor.
      std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>
          matmul_fwd_pd = matmul_fwd->GetPrimitiveDesc();
      this->AllocateOutputTensor(context, *matmul_fwd_pd, dst_dims_mkl_order,
                                 input_output_fmt_mkldnn, &dst_tensor);

      Toutput* dst_data =
          reinterpret_cast<Toutput*>(dst_tensor->flat<Toutput>().data());

      // Check if src and weight data need to be reordered.
      Tinput* src_data = nullptr;
      if (IS_SRC_REORDER_NEEDED(src_md, matmul_fwd_pd, matmul_fwd)) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(
            MEMORY_PD_WITHOUT_DATA(matmul_fwd_pd.get()->PRIMITIVE_DESC_SRC,
                                   this->cpu_engine_),
            context);
        src_data = static_cast<Tinput*>(src.GetOpMem().get_data_handle());
      } else {
        src_data = static_cast<Tinput*>(
            const_cast<Tinput*>(src_tensor.flat<Tinput>().data()));
      }

      Tweight* weight_data = nullptr;
      if (IS_WEIGHTS_REORDER_NEEDED(weight_md, matmul_fwd_pd, matmul_fwd)) {
        bool is_weight_cached = false;
        // For batch size 1, MKL-DNN expects that weight format is OI whereas
        // TF default format is IO. So in that case convert weight from IO
        // to OI for the first iteration and cache it to reuse in the
        // subsequent iterations, if the weight is constant.
        if (this->is_weight_const_) {
          // Check if the weight is already cached or not
          if (this->IsWeightCacheEmpty(context)) {
            // Cache weight if it is not cached.
            this->CacheWeight(context, matmul_fwd_pd, weight_data,
                              weight_tensor, weight, weight_md);
          }
#ifdef ENABLE_MKLDNN_V1
          weight_data = this->GetCachedWeight(
              context, GET_WEIGHTS_DESC_FROM_OP_PD(matmul_fwd_pd));
#else
          weight_data = this->GetCachedWeight(
              context, GET_WEIGHTS_DESC_FROM_OP_PD(matmul_fwd_pd).desc());
#endif
          is_weight_cached = (weight_data != nullptr);
        }

        if (!is_weight_cached) {
          weight.SetUsrMem(weight_md, &weight_tensor);
          weight.CheckReorderToOpMem(
              MEMORY_PD_WITHOUT_DATA(
                  matmul_fwd_pd.get()->PRIMITIVE_DESC_WEIGHTS,
                  this->cpu_engine_),
              context);
          weight_data =
              static_cast<Tweight*>(weight.GetOpMem().get_data_handle());
        }

      } else {
        weight_data = static_cast<Tweight*>(
            const_cast<Tweight*>(weight_tensor.flat<Tweight>().data()));
      }

      std::shared_ptr<stream> cpu_stream;
      cpu_stream.reset(CreateStream(context, matmul_fwd->GetEngine()));
      // Execute inner-product
      Tbias* bias_data = this->GetBiasHandle(
          context, matmul_fwd_pd, bias_tensor, weight_tensor, cpu_stream);
      matmul_fwd->Execute(src_data, weight_data, bias_data, dst_data,
                          cpu_stream);
    } catch (mkldnn::error& e) {
      string error_msg = tensorflow::strings::StrCat(
          "Status: ", e.status, ", message: ", string(e.message), ", in file ",
          __FILE__, ":", __LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
    float min_output_value;
    float max_output_value;
    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value) {
      // This is the case the inner-product and requantization are fused.
      // "min_freezed_output" and "max_freezed_output" are the requested range
      // for the output.
      min_output_value = context->input(7).flat<float>()(0);
      max_output_value = context->input(8).flat<float>()(0);
    } else {
      ComputeOutputRangeForInt32(context, &min_output_value, &max_output_value);
    }

    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, qint32>::value) {
      Tensor* output_min = nullptr;
      Tensor* output_max = nullptr;
      MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
      output_min_mkl_shape.SetMklTensor(false);
      output_max_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 1, &output_min, {},
                                output_min_mkl_shape);
      AllocateOutputSetMklShape(context, 2, &output_max, {},
                                output_max_mkl_shape);
      output_min->flat<float>()(0) = min_output_value;
      output_max->flat<float>()(0) = max_output_value;
    }
  }

 protected:
  void ComputeOutputRangeForInt32(OpKernelContext* context,
                                  float* min_output_value,
                                  float* max_output_value) {
    const float min_input = context->input(3).flat<float>()(0);
    const float max_input = context->input(4).flat<float>()(0);
    const float min_weight = context->input(5).flat<float>()(0);
    const float max_weight = context->input(6).flat<float>()(0);
    MklQuantizationRangeForMultiplication<quint8, qint8, qint32>(
        min_input, max_input, min_weight, max_weight, min_output_value,
        max_output_value);
  }

  virtual void ExtendMklDnnMatMulFwdParams(OpKernelContext* context,
                                           MklDnnMatMulFwdParams& params) {
    // Append data type names of input, weight, bias, and output.
    params.dtypes.append(typeid(Tinput).name());
    params.dtypes.append(typeid(Tweight).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());

    // When the output type is quint8, the output data is requantized into
    // quint8. A post_op "output_scale" is added to do the conversion.
    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, float>::value) {
      float min_output_value;
      float max_output_value;
      ComputeOutputRangeForInt32(context, &min_output_value, &max_output_value);
      float scale_int32 =
          std::max(std::abs(min_output_value), std::abs(max_output_value));
      const float min_freezed_output = context->input(7).flat<float>()(0);
      const float max_freezed_output = context->input(8).flat<float>()(0);
      float scale_eightbit =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      float scale = 1.0;
      if (std::is_same<Toutput, quint8>::value) {
        scale = scale_int32 / scale_eightbit / static_cast<float>(1u << 23);
      } else if (std::is_same<Toutput, qint8>::value) {
        scale = scale_int32 / scale_eightbit / static_cast<float>(1u << 24);
      } else if (std::is_same<Toutput, float>::value) {
        scale = scale_int32 / static_cast<float>(1u << 31);
      } else {
        // @TODO:keeping the default qint8 as before. Change to error later.
        scale = scale_int32 / scale_eightbit / static_cast<float>(1u << 24);
      }
      std::vector<float> output_scale;
      output_scale.push_back(scale);
      params.post_op_params.push_back({"output_scale", output_scale});
    }
  }

  // This function handles bias conversion and compensation for MIN_FIRST and
  // SCALE mode. If input is quantized via MIN_FIRST,
  //  B's32 = Q'a * Qw * Bf32 + Q'a * Qw * Min(Af32) * 1 * Wf32
  // If input is quantized via SCALE,
  //   Bs32 = Qa * Qw * Bf32.
  Tbias* GetBiasHandle(
      OpKernelContext* context,
      std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>&
          mkldnn_matmul_fwd_pd,
      const Tensor& bias_tensor, const Tensor& weight_tensor,
      std::shared_ptr<stream> reorder_stream) {
    // If the bias is qint32, it means the bias is already converted offline.
    // and it can be added to matmul output directly.
    if (std::is_same<Tbias, qint32>::value) {
      return static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
    } else {
      // If the bias is fp32, then need to calculate the bias
      const float min_input = context->input(3).flat<float>()(0);
      const float max_input = context->input(4).flat<float>()(0);
      const float min_weight = context->input(5).flat<float>()(0);
      const float max_weight = context->input(6).flat<float>()(0);

      std::vector<mkldnn::primitive> net;
      float out_scale;
      // If the bias is float and input quantize is MIN_FIRST, bias has to be
      // compensated with B's32 = Q'a * Qw * Bf32 + Q'a * Qw * Min(Af32) * 1 *
      // Wf32.
      if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
        int k = weight_tensor.dim_size(0);
        int n = weight_tensor.dim_size(1);
        float* comp_bias = GetCompBiasBuffer(n);

        qint8* wt_buf = static_cast<qint8*>(
            const_cast<qint8*>(weight_tensor.flat<qint8>().data()));

        const float* bias_buf = static_cast<float*>(
            const_cast<float*>(bias_tensor.flat<float>().data()));

        float qa_amin = 255 * min_input / (max_input - min_input);

        out_scale = (255.0 * 127.0) /
                    ((max_input - min_input) *
                     std::max(std::abs(max_weight), std::abs(min_weight)));

#pragma omp parallel for schedule(static)
        for (int j = 0; j < n; ++j) {
          int x = 0;
          for (int i = 0; i < k; ++i) {
            x += wt_buf[i * n + j];
          }
          comp_bias[j] =
              ((bias_buf[j] * out_scale) + static_cast<float>(x * qa_amin));
        }

        return reinterpret_cast<Tbias*>(comp_bias_);

      } else if (mode_ == QUANTIZE_MODE_SCALED) {
        // If the bias is float and input quantize is SCALE, bias has to be
        // compensated with Bs32 = Qa * Qw * Bf32.
        out_scale = 255.0 * 127.0 / max_input *
                    std::max(std::abs(max_weight), std::abs(min_weight));

        std::vector<float> scales;
        scales.push_back(out_scale);
        mkldnn::primitive_attr bias_attr;
        bias_attr.set_output_scales(0, scales);

        void* bias_buf = static_cast<void*>(
            const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
        input_bias_ =
            new MEMORY_CONSTRUCTOR(mkldnn_matmul_fwd_pd->PRIMITIVE_DESC_BIAS,
                                   this->cpu_engine_, bias_buf);
        scaled_bias_ = new MEMORY_CONSTRUCTOR_WITHOUT_DATA(
            mkldnn_matmul_fwd_pd->PRIMITIVE_DESC_BIAS, this->cpu_engine_);

#ifdef ENABLE_MKLDNN_V1
        auto reorder_desc = mkldnn::reorder::primitive_desc(
            *input_bias_, *scaled_bias_, bias_attr);
        net.push_back(mkldnn::reorder(reorder_desc));
        std::unordered_map<int, memory> reorder_net_args = {
            {MKLDNN_ARG_FROM, *input_bias_},
            { MKLDNN_ARG_TO,
              *scaled_bias_ }};
        net.at(0).execute(*reorder_stream, reorder_net_args);
#else
        auto reorder_desc = mkldnn::reorder::primitive_desc(
            input_bias_->get_primitive_desc(),
            scaled_bias_->get_primitive_desc(), bias_attr);
        net.push_back(
            mkldnn::reorder(reorder_desc, *input_bias_, *scaled_bias_));
        reorder_stream->submit(net).wait();
#endif  // ENABLE_MKLDNN_V1

        return reinterpret_cast<Tbias*>(scaled_bias_->get_data_handle());
      } else {
        context->CtxFailure(
            errors::InvalidArgument("Quantization mode must be"
                                    "either MIN_FIRST or SCALED."));
        return nullptr;
      }
    }
  }

 private:
  memory* input_bias_ = nullptr;
  memory* scaled_bias_ = nullptr;

  // Buffer to save the compensated bias
  float* comp_bias_ = nullptr;

  int mode_;
};

template <typename Device, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnQuantizedMatMulReluOp
    : public MklDnnQuantizedMatMulOp<Device, Tinput, Tweight, Tbias, Toutput> {
 public:
  virtual ~MklDnnQuantizedMatMulReluOp() {}

  explicit MklDnnQuantizedMatMulReluOp(OpKernelConstruction* context)
      : MklDnnQuantizedMatMulOp<Device, Tinput, Tweight, Tbias, Toutput>(
            context) {}

 protected:
  void ExtendMklDnnMatMulFwdParams(OpKernelContext* context,
                                   MklDnnMatMulFwdParams& params) override {
    MklDnnQuantizedMatMulOp<Device, quint8, qint8, Tbias,
                            Toutput>::ExtendMklDnnMatMulFwdParams(context,
                                                                  params);
    params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
  }
};

// Register NoOp kernel for QuantizedMatMulWithBias to get a python interface.
// This kernel will be replaced by an MKL kernel during graph
// optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedMatMulWithBias")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint<qint32>("Toutput"),
                        NoOp);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBias")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<qint32>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulOp<CPUDevice, quint8, qint8, float, qint32>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBias")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<qint32>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulOp<CPUDevice, quint8, qint8, qint32, qint32>);

// Register NoOp kernel for QuantizedMatMulWithBiasAndRelu to get a python
// interface. This kernel will be replaced by an MKL kernel during
// graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedMatMulWithBiasAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint<qint32>("Toutput"),
                        NoOp);
// Register NoOp kernel for QuantizedIPWithBiasAndReluAndRequantize
// to get a python interface. This kernel will be replaced by an MKL kernel
// during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedMatMulWithBiasAndReluAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint("Tbias", {DT_QINT32, DT_FLOAT})
                            .TypeConstraint<quint8>("Toutput"),
                        NoOp);

// Register NoOp kernel for QuantizedMatMulWithBiasAndRequantize
// to get a python interface. This kernel will be replaced by an MKL kernel
// during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedMatMulWithBiasAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint("Tbias", {DT_QINT32, DT_FLOAT})
                            .TypeConstraint<quint8>("Toutput"),
                        NoOp);

// Register NoOp kernel for QuantizedMatMulWithBiasAndDequantize
// to get a python interface. This kernel will be replaced by an MKL kernel
// during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedMatMulWithBiasAndDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint("Tbias", {DT_QINT32, DT_FLOAT})
                            .TypeConstraint<float>("Toutput"),
                        NoOp);

// Register a templatized implementation of _MklQuantizedMatMulWithBiasAndRelu.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBiasAndRelu")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<qint32>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulReluOp<CPUDevice, quint8, qint8, float, qint32>);
// Register a templatized implementation of
// _MklQuantizedMatMulWithBiasAndReluAndRequantize.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulReluOp<CPUDevice, quint8, qint8, qint32, quint8>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulReluOp<CPUDevice, quint8, qint8, float, quint8>);

// Register a templatized implementation of
// _MklQuantizedMatMulWithBiasAndRequantize.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulOp<CPUDevice, quint8, qint8, qint32, quint8>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulOp<CPUDevice, quint8, qint8, float, quint8>);

// Register a templatized implementation of
// _MklQuantizedMatMulWithBiasAndDequantize.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBiasAndDequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<float>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulOp<CPUDevice, quint8, qint8, qint32, float>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedMatMulWithBiasAndDequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<float>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklDnnQuantizedMatMulOp<CPUDevice, quint8, qint8, float, float>);

}  // namespace tensorflow

#endif  // INTEL_MKL
