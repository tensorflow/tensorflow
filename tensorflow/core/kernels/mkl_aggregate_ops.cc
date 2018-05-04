/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include <numeric>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

#ifndef INTEL_MKL_ML
#include "mkldnn.hpp"
using mkldnn::stream;
using mkldnn::sum;
#endif

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

#ifdef INTEL_MKL_ML

template <typename Device, typename T>
class MklAddNOp : public OpKernel {
 public:
  explicit MklAddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const int num = ctx->num_inputs();
    OP_REQUIRES(ctx, num / 2 == 2,
                errors::InvalidArgument("Only additions of two tensors "
                                        "supported by MKL. Num inputs: ",
                                        num));

    MklAddNOpContext mkl_context;
    size_t src1_idx = 0, src2_idx = 1;
    const Tensor& input0 = MklGetInput(ctx, src1_idx);
    GetMklShape(ctx, src1_idx, &(mkl_context.input1_shape));
    bool input1_in_mkl_format = mkl_context.input1_shape.IsMklTensor();

    const Tensor& input1 = MklGetInput(ctx, src2_idx);
    GetMklShape(ctx, src2_idx, &(mkl_context.input2_shape));
    bool input2_in_mkl_format = mkl_context.input2_shape.IsMklTensor();

    // if the shapes of two tensors are not same raise op error
    TensorShape src1_shape, src2_shape;
    src1_shape = input0.shape();
    src2_shape = input1.shape();
    if (!src1_shape.IsSameSize(src2_shape)) {
      ctx->SetStatus(errors::InvalidArgument(
          "Inputs to operation ", this->name(), " of type ",
          this->type_string(), " must have the same size and shape.  Input 0: ",
          src1_shape.DebugString(), " != input 1: ", src2_shape.DebugString()));
    }
    // handle the case of a scalar
    if (!input1_in_mkl_format && input0.dims() == 0) {
      const TensorShape& o_shape = input0.shape();
      Tensor* out_tensor = nullptr;
      mkl_context.output_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(ctx, src1_idx, &out_tensor, o_shape,
                                mkl_context.output_shape);
      float user_i1 = (input0.scalar<T>()());
      float user_i2 = (input1.scalar<T>()());
      out_tensor->scalar<T>()() = std::plus<float>{}(user_i1, user_i2);
      return;
    }

    mkl_context.in_dims = input1_in_mkl_format
                              ? mkl_context.input1_shape.GetDimension()
                              : input0.dims();
    mkl_context.in_dims = input2_in_mkl_format
                              ? mkl_context.input2_shape.GetDimension()
                              : input1.dims();

    // If there is nothing to compute, return.
    if (!input1_in_mkl_format && !input2_in_mkl_format) {
      const TensorShape& o_shape = input0.shape();
      if (o_shape.num_elements() == 0) {
        Tensor* out_tensor = nullptr;
        mkl_context.output_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(ctx, src1_idx, &out_tensor, o_shape,
                                  mkl_context.output_shape);
        return;
      }
    }

    mkl_context.in_sizes = new size_t[mkl_context.in_dims];
    mkl_context.in_strides = new size_t[mkl_context.in_dims];
    // Generate size, stride for input if input is in MKL format.
    if (input1_in_mkl_format || input2_in_mkl_format) {
      const MklShape* tmp_mkl_shape = (input1_in_mkl_format)
                                          ? &mkl_context.input1_shape
                                          : &mkl_context.input2_shape;
      for (int i = 0; i < mkl_context.in_dims; i++) {
        mkl_context.in_sizes[i] = tmp_mkl_shape->GetSizes()[i];
        mkl_context.in_strides[i] = tmp_mkl_shape->GetStrides()[i];
      }
    } else {
      for (int i = 0; i < mkl_context.in_dims; i++) {
        mkl_context.in_sizes[i] =
            input0.dim_size((mkl_context.in_dims - 1) - i);
      }
      mkl_context.in_strides[0] = 1;
      for (int i = 1; i < mkl_context.in_dims; i++) {
        mkl_context.in_strides[i] =
            mkl_context.in_strides[i - 1] * mkl_context.in_sizes[i - 1];
      }
    }
    std::vector<float> coeff(2, 1.0);
    mkl_context.MklCreateInputLayouts(ctx);
    CHECK_EQ(dnnSumCreate_F32(&mkl_context.Eltwise, mkl_context.attributes, 2,
                              mkl_context.lt_input1, &coeff[0]),
             E_SUCCESS);

    Tensor mkl_tmp_input1_buf_tensor, mkl_tmp_input2_buf_tensor;
    mkl_context.MklPrepareAddNInputs(ctx, &mkl_tmp_input1_buf_tensor,
                                     &mkl_tmp_input2_buf_tensor);
    Tensor* output = nullptr;
    if (input1_in_mkl_format || input2_in_mkl_format) {
      TensorShape tf_shape;
      mkl_context.output_shape.SetMklTensor(true);
      mkl_context.output_shape.SetMklLayout(mkl_context.Eltwise,
                                            dnnResourceDst);

      mkl_context.output_shape.SetTfLayout(
          mkl_context.in_dims, mkl_context.in_sizes, mkl_context.in_strides);
      if (input1_in_mkl_format == true) {
        mkl_context.output_shape.SetTfDimOrder(
            mkl_context.in_dims, mkl_context.input1_shape.GetTfToMklDimMap());
      } else {
        mkl_context.output_shape.SetTfDimOrder(
            mkl_context.in_dims, mkl_context.input2_shape.GetTfToMklDimMap());
      }
      tf_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                          mkl_context.output_shape.GetMklLayout())) /
                      sizeof(T));

      AllocateOutputSetMklShape(ctx, src1_idx, &output, tf_shape,
                                mkl_context.output_shape);
    } else {
      const TensorShape& o_shape = input1.shape();
      mkl_context.output_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(ctx, src1_idx, &output, o_shape,
                                mkl_context.output_shape);
    }

    mkl_context.Eltwise_res[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    // Execute convolution
    CHECK_EQ(dnnExecute_F32(mkl_context.Eltwise, mkl_context.Eltwise_res),
             E_SUCCESS);

    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    int in_dims;
    size_t* in_sizes = nullptr;
    size_t* in_strides = nullptr;
    dnnPrimitive_t Eltwise = nullptr;
    dnnPrimitiveAttributes_t attributes = nullptr;
    void* Eltwise_res[dnnResourceNumber];
    dnnLayout_t lt_input1 = nullptr, lt_input2 = nullptr;
    MklShape input1_shape, input2_shape, output_shape;

    void MklCreateInputLayouts(OpKernelContext* context) {
      bool input1_in_mkl_format = input1_shape.IsMklTensor();
      if (!input1_in_mkl_format) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input1, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      } else {
        lt_input1 = static_cast<dnnLayout_t>(input1_shape.GetCurLayout());
      }

      bool input2_in_mkl_format = input2_shape.IsMklTensor();
      if (!input2_in_mkl_format) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input2, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      } else {
        lt_input2 = static_cast<dnnLayout_t>(input2_shape.GetCurLayout());
      }
    }

    void MklPrepareAddNInputs(OpKernelContext* context,
                              Tensor* mkl_tmp_input1_buf_tensor,
                              Tensor* mkl_tmp_input2_buf_tensor) {
      bool mkl_convert_input1, mkl_convert_input2;
      dnnPrimitive_t mkl_prim_convert_input1 = nullptr,
                     mkl_prim_convert_input2 = nullptr;
      dnnLayout_t mkl_lt_internal_input1 = nullptr,
                  mkl_lt_internal_input2 = nullptr;
      void *mkl_buf_convert_input1 = nullptr, *mkl_buf_convert_input2 = nullptr;
      dnnResourceType_t dnnResourceMultipleSrc2 =
          (dnnResourceType_t)(dnnResourceMultipleSrc + 1);
      // Compare with internal layouts and convert if needed
      const Tensor& input1 = MklGetInput(context, 0);

      void* mkl_buf_input1 =
          const_cast<void*>(static_cast<const void*>(input1.flat<T>().data()));

      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &mkl_lt_internal_input1, Eltwise, dnnResourceMultipleSrc),
               E_SUCCESS);
      mkl_convert_input1 =
          !dnnLayoutCompare_F32(mkl_lt_internal_input1, lt_input1);
      if (mkl_convert_input1) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input1, lt_input1,
                                         mkl_lt_internal_input1),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_input1_buf_tensor,
                       mkl_lt_internal_input1, &mkl_buf_convert_input1);
        CHECK_EQ(
            dnnConversionExecute_F32(mkl_prim_convert_input1, mkl_buf_input1,
                                     mkl_buf_convert_input1),
            E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_input1);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_input1);

      Eltwise_res[dnnResourceMultipleSrc] =
          (mkl_convert_input1) ? mkl_buf_convert_input1 : mkl_buf_input1;

      const Tensor& input2 = MklGetInput(context, 1);
      void* mkl_buf_input2 =
          const_cast<void*>(static_cast<const void*>(input2.flat<T>().data()));
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &mkl_lt_internal_input2, Eltwise, dnnResourceMultipleSrc2),
               E_SUCCESS);
      mkl_convert_input2 =
          !dnnLayoutCompare_F32(mkl_lt_internal_input2, lt_input2);
      if (mkl_convert_input2) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input2, lt_input2,
                                         mkl_lt_internal_input2),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_input2_buf_tensor,
                       mkl_lt_internal_input2, &mkl_buf_convert_input2);
        CHECK_EQ(
            dnnConversionExecute_F32(mkl_prim_convert_input2, mkl_buf_input2,
                                     mkl_buf_convert_input2),
            E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_input2);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_input2);

      Eltwise_res[dnnResourceMultipleSrc2] =
          (mkl_convert_input2) ? mkl_buf_convert_input2 : mkl_buf_input2;
    }

    void MklCleanup() {
      bool input1_in_mkl_format = input1_shape.IsMklTensor();
      bool input2_in_mkl_format = input2_shape.IsMklTensor();
      dnnDelete_F32(Eltwise);
      if (!input1_in_mkl_format || !input2_in_mkl_format) {
        delete[] in_sizes;
        delete[] in_strides;
      }
      if (!input1_in_mkl_format) {
        dnnLayoutDelete_F32(lt_input1);
      }
      if (!input2_in_mkl_format) {
        dnnLayoutDelete_F32(lt_input2);
      }
    }
  } MklAddNOpContext;
};

#else  // INTEL_MKL_ML
template <typename Device, typename T>
class MklAddNOp : public OpKernel {
 public:
  ~MklAddNOp() {}
  explicit MklAddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const int num = ctx->num_inputs();
    // Only additions of 2 input tensors is supported now
    OP_REQUIRES(ctx, num / 2 == 2,
                errors::InvalidArgument("Only additions of two tensors "
                                        "supported by MKL. Num inputs: ",
                                        num));

    try {
      auto cpu_engine = engine(engine::cpu, 0);
      size_t src1_idx = 0, src2_idx = 1, output_idx = 0;
      const Tensor& src1_tensor = MklGetInput(ctx, src1_idx);
      const Tensor& src2_tensor = MklGetInput(ctx, src2_idx);

      MklDnnShape src1_mkl_shape, src2_mkl_shape;
      GetMklShape(ctx, src1_idx, &src1_mkl_shape);
      GetMklShape(ctx, src2_idx, &src2_mkl_shape);
      bool input1_in_mkl_format = src1_mkl_shape.IsMklTensor();
      bool input2_in_mkl_format = src2_mkl_shape.IsMklTensor();
      int src1_dims_size = input1_in_mkl_format ? src1_mkl_shape.GetDimension()
                                                : src1_tensor.dims();
      int src2_dims_size = input2_in_mkl_format ? src2_mkl_shape.GetDimension()
                                                : src2_tensor.dims();
      // if the shapes of two tensors are not same raise op error
      TensorShape src1_shape, src2_shape;
      src1_shape = input1_in_mkl_format ? src1_mkl_shape.GetTfShape()
                                        : src1_tensor.shape();
      src2_shape = input2_in_mkl_format ? src2_mkl_shape.GetTfShape()
                                        : src2_tensor.shape();

      if (!src1_shape.IsSameSize(src2_shape)) {
        ctx->SetStatus(errors::InvalidArgument(
            "Inputs to operation ", this->name(), " of type ",
            this->type_string(),
            " must have the same size and shape.  Input 0: ",
            src1_shape.DebugString(),
            " != input 1: ", src2_shape.DebugString()));
      }

      if (!input1_in_mkl_format && src1_dims_size == 0) {
        Tensor* dst_tensor = nullptr;
        MklShape mkl_shape_dst;
        mkl_shape_dst.SetMklTensor(false);
        AllocateOutputSetMklShape(ctx, output_idx, &dst_tensor,
                                  src1_tensor.shape(), mkl_shape_dst);
        float user_i1 = (src1_tensor.scalar<T>()());
        float user_i2 = (src2_tensor.scalar<T>()());
        dst_tensor->scalar<T>()() = std::plus<float>{}(user_i1, user_i2);
        return;
      }

      // If there is nothing to compute, return.
      if (!input1_in_mkl_format && !input2_in_mkl_format) {
        if (src1_tensor.shape().num_elements() == 0) {
          Tensor* dst_tensor = nullptr;
          MklShape mkl_shape_dst;
          mkl_shape_dst.SetMklTensor(false);
          AllocateOutputSetMklShape(ctx, output_idx, &dst_tensor,
                                    src1_tensor.shape(), mkl_shape_dst);
          return;
        }
      }

      std::vector<double> coeff(2, 1.0);
      MklDnnData<T> src1(&cpu_engine);
      MklDnnData<T> src2(&cpu_engine);
      MklDnnData<T> dst(&cpu_engine);

      int tmp_size = input1_in_mkl_format ? src2_dims_size : src1_dims_size;
      memory::dims dims(tmp_size);
      memory::dims strides(tmp_size);
      memory::desc md1({}, memory::data_undef, memory::format_undef);
      memory::desc md2({}, memory::data_undef, memory::format_undef);

      // For creating Sum primitive, we need to ensure that all inputs are in
      // same format. What that means is if we have a mixed input case - where
      // one input is in Tensorflow format and one input is in MKL format -,
      // then we need to ensure that all inputs are in same format for
      // primitive construction. For performance reason, we say that all inputs
      // are in MKL format in such case, and insert reorder for input that is
      // in Tensorflow format into MKL format. On the other hand, if both the
      // inputs are in MKL format or both are in Tensorflow format, then we
      // dont need reorder.
      if (!input1_in_mkl_format && !input2_in_mkl_format) {
        // If both the inputs are in Tensorflow format, we create blocked memory
        // descriptor.
        dims = TFShapeToMklDnnDims(src1_tensor.shape());
        strides = CalculateTFStrides(dims);
        md1 = MklDnnData<T>::CreateBlockedMemDesc(dims, strides);
        md2 = md1;
      } else if (input1_in_mkl_format && !input2_in_mkl_format) {
        // If one input is in MKL format and other is in Tensorflow, then
        // create respective descriptors describing the actual case. For input
        // in Mkl format, we just get Mkl layout from MklDnnShape. For input in
        // Tensorflow format, we create memory descriptor using data format.
        md1 = src1_mkl_shape.GetMklLayout();

        memory::format src1_mkl_data_format = src1_mkl_shape.GetTfDataFormat();
        auto src1_tf_data_format =
            MklDnnDataFormatToTFDataFormat(src1_mkl_data_format);
        auto src2_dims =
            TFShapeToMklDnnDimsInNCHW(src2_tensor.shape(), src1_tf_data_format);
        md2 = memory::desc(src2_dims, MklDnnType<T>(), src1_mkl_data_format);
      } else if (input2_in_mkl_format && !input1_in_mkl_format) {
        // Same comment as above.
        memory::format src2_mkl_data_format = src2_mkl_shape.GetTfDataFormat();
        auto src2_tf_data_format =
            MklDnnDataFormatToTFDataFormat(src2_mkl_data_format);
        auto src1_dims =
            TFShapeToMklDnnDimsInNCHW(src1_tensor.shape(), src2_tf_data_format);
        md1 = memory::desc(src1_dims, MklDnnType<T>(), src2_mkl_data_format);

        md2 = src2_mkl_shape.GetMklLayout();
      } else {
        // If both the inputs are in MKL format, we use Mkl layout of the input
        // tensors.
        md1 = src1_mkl_shape.GetMklLayout();
        md2 = src2_mkl_shape.GetMklLayout();
      }
      src1.SetUsrMem(md1, &src1_tensor);
      src2.SetUsrMem(md2, &src2_tensor);

      // As per comment above, we tell MKLDNN that both the inputs are in same
      // format. So we set common memory descriptor in MKL format, if any of the
      // inputs are in MKL format. Let's get memory descriptor that we will use
      // for both the inputs.
      // We set output memory descriptor in MKL format, if any of the
      // inputs are in MKL format.
      memory::desc common_md({}, memory::data_undef, memory::format_undef);
      if (input1_in_mkl_format || input2_in_mkl_format) {
        common_md = input1_in_mkl_format ? md1 : md2;
        dst.SetUsrMem(common_md);
      } else {
        // Since both the inputs are in Tensorflow format, and have
        // same shape, we can get memory descriptor from any input.
        common_md = md1;
        dst.SetUsrMem(common_md);
      }

      std::vector<memory::primitive_desc> srcs_pd;
      // Memory descriptor for 1st input
      srcs_pd.push_back(memory::primitive_desc(common_md, cpu_engine));
      // Memory descriptor for 2nd input
      srcs_pd.push_back(memory::primitive_desc(common_md, cpu_engine));
      auto sum_pd = sum::primitive_desc(dst.GetUsrMemDesc(), coeff, srcs_pd);

      // Now we setup resources for primitive execution.
      // First, we need to check if any of the inputs need to be reordered as
      // per the logic described above. Since output will be in MKL format if
      // atleast one input is in MKL format, we choose output descriptor for
      // reorder.
      std::vector<primitive::at> inputs;
      std::vector<primitive> net;
      // Check if actual input format of the tensor is different than common_pd
      // we told MKLDNN. In that case, we will need reorder.
      src1.CheckReorderToOpMem(srcs_pd[0], &net);
      src2.CheckReorderToOpMem(srcs_pd[1], &net);
      inputs.push_back(src1.GetOpMem());
      inputs.push_back(src2.GetOpMem());

      // Allocate output tensor now.
      Tensor* dst_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;

      if (input2_in_mkl_format || input1_in_mkl_format) {
        output_mkl_shape.SetMklTensor(true);
        auto output_pd = dst.GetUsrMemPrimDesc();
        output_mkl_shape.SetMklLayout(&output_pd);
        output_mkl_shape.SetElemType(MklDnnType<T>());
        if (input1_in_mkl_format) {
          output_mkl_shape.SetTfLayout(src1_dims_size,
                                       src1_mkl_shape.GetSizesAsMklDnnDims(),
                                       src1_mkl_shape.GetTfDataFormat());
        } else {
          output_mkl_shape.SetTfLayout(src2_dims_size,
                                       src2_mkl_shape.GetSizesAsMklDnnDims(),
                                       src2_mkl_shape.GetTfDataFormat());
        }
        output_tf_shape.AddDim((output_pd.get_size() / sizeof(T)));
      } else {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = src1_tensor.shape();
      }
      AllocateOutputSetMklShape(ctx, output_idx, &dst_tensor, output_tf_shape,
                                output_mkl_shape);
      dst.SetUsrMemDataHandle(dst_tensor);

      // Create Sum op, and submit net for execution.
      net.push_back(sum(sum_pd, inputs, dst.GetOpMem()));
      stream(stream::kind::eager).submit(net).wait();
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }
};

#endif
#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklAddN")                          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklAddNOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow
#endif  // INTEL_MKL
