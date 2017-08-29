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

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklAddNOp : public OpKernel {
 public:
  explicit MklAddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const int num = ctx->num_inputs();
    OP_REQUIRES(ctx, num / 2 == 2,
                errors::InvalidArgument("Only additions of two arguments "
                                        "supported by MKL. Num inputs: ",
                                        num));

    MklAddNOpContext mkl_context;
    const Tensor& input0 = MklGetInput(ctx, 0);
    GetMklShape(ctx, 0, &(mkl_context.input1_shape));
    bool input1_in_mkl_format = mkl_context.input1_shape.IsMklTensor();

    const Tensor& input1 = MklGetInput(ctx, 1);
    GetMklShape(ctx, 1, &(mkl_context.input2_shape));
    bool input2_in_mkl_format = mkl_context.input2_shape.IsMklTensor();

    mkl_context.in_dims = input1_in_mkl_format
        ? mkl_context.input1_shape.GetDimension()
        : input0.dims();
    mkl_context.in_dims = input2_in_mkl_format
        ? mkl_context.input2_shape.GetDimension()
        : input1.dims();
    // Generate size, stride for input if input is in MKL format.
    ExtractMklOpParams(&mkl_context.in1_sizes,
     &mkl_context.in1_strides, input0, &mkl_context.input1_shape);
    ExtractMklOpParams(&mkl_context.in2_sizes,
     &mkl_context.in2_strides, input1, &mkl_context.input2_shape);

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
     mkl_context.output_shape.SetMklLayout(mkl_context.Eltwise, dnnResourceDst);

     mkl_context.output_shape.SetTfLayout(
        mkl_context.in_dims, mkl_context.in1_sizes, mkl_context.in1_strides);
     if (input1_in_mkl_format == true) {
      mkl_context.output_shape.SetTfDimOrder(mkl_context.in_dims,
      mkl_context.input1_shape.GetTfToMklDimMap());
     } else {
      mkl_context.output_shape.SetTfDimOrder(mkl_context.in_dims,
      mkl_context.input2_shape.GetTfToMklDimMap());
     }
     tf_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                        mkl_context.output_shape.GetMklLayout())) /
                    sizeof(T));

     AllocateOutputSetMklShape(ctx, 0, &output, tf_shape,
                              mkl_context.output_shape);
    } else {
     const TensorShape& o_shape = input1.shape();
     mkl_context.output_shape.SetMklTensor(false);
     AllocateOutputSetMklShape(ctx, 0, &output, o_shape,
                                mkl_context.output_shape);
    }

    mkl_context.Eltwise_res[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    // Execute convolution
    CHECK_EQ(dnnExecute_F32(mkl_context.Eltwise, mkl_context.Eltwise_res),
             E_SUCCESS);

    mkl_context.MklCleanup();
  }

  void ExtractMklOpParams(size_t** out_sizes, size_t** out_strides,
    const Tensor& input, const MklShape* input_shape) {
    bool input_in_mkl_format = input_shape->IsMklTensor();
    int in_dims = input_in_mkl_format
                              ? input_shape->GetDimension()
                              : input.dims();
    size_t* in_sizes = new size_t[in_dims];
    size_t* in_strides = new size_t[in_dims];

    if (input_in_mkl_format) {
      for (int i = 0; i < in_dims; i++) {
        in_sizes[i] = input_shape->GetSizes()[i];
        in_strides[i] = input_shape->GetStrides()[i];
      }
    } else {
      for (int i = 0; i < in_dims; i++) {
        in_sizes[i] =
            input.dim_size((in_dims - 1) - i);
      }
      in_strides[0] = 1;
      for (int i = 1; i < in_dims; i++) {
        in_strides[i] =
            in_strides[i - 1] * in_sizes[i - 1];
      }
    }
    *out_sizes = in_sizes;
    *out_strides = in_strides;
  }


 private:
  typedef struct {
    int in_dims;
    size_t* in1_sizes;
    size_t* in1_strides;

    size_t* in2_sizes;
    size_t* in2_strides;
    dnnPrimitive_t Eltwise = nullptr;
    dnnPrimitiveAttributes_t attributes = nullptr;
    void* Eltwise_res[dnnResourceNumber];
    dnnLayout_t lt_input1 = nullptr, lt_input2 = nullptr;
    MklShape input1_shape, input2_shape, output_shape;

    void MklCreateInputLayouts(OpKernelContext* context) {
      bool input1_in_mkl_format = input1_shape.IsMklTensor();
      if (!input1_in_mkl_format) {
        CHECK_EQ(
            dnnLayoutCreate_F32(&lt_input1, in_dims, in1_sizes, in1_strides),
            E_SUCCESS);
      } else {
        lt_input1 = static_cast<dnnLayout_t>(input1_shape.GetCurLayout());
      }

      bool input2_in_mkl_format = input2_shape.IsMklTensor();
      if (!input2_in_mkl_format) {
        CHECK_EQ(
            dnnLayoutCreate_F32(&lt_input2, in_dims, in2_sizes, in2_strides),
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
      if (!input1_in_mkl_format) {
         dnnLayoutDelete_F32(lt_input1);
         delete [] in1_sizes;
         delete [] in1_strides;
      }
      if (!input2_in_mkl_format) {
         dnnLayoutDelete_F32(lt_input2);
         delete [] in2_sizes;
         delete [] in2_strides;
      }
    }
  } MklAddNOpContext;
};

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
