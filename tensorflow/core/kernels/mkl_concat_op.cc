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

#ifdef INTEL_MKL

#include <limits>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

// TODO(intelft) Check if we can reuse existing EigenConcatOp using Mutable
// reference inputs.
// --------------------------------------------------------------------------
//                      Eigen Concat Op
// --------------------------------------------------------------------------
template <typename Device, typename T, AxisArgumentName AxisArgName>
class EigenConcatBaseOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit EigenConcatBaseOp(OpKernelConstruction* c) : OpKernel(c) {}

  // Although, we modify Compute for this call to accept one extra param,
  // we need to have empty Compute because Compute is pure virtual function.
  void Compute(OpKernelContext* c) {}

  void Compute(OpKernelContext* c, const std::vector<Tensor>& values) {
    const Tensor* concat_dim_tensor;
    const char* axis_attribute_name =
        AxisArgName == NAME_IS_AXIS
            ? "axis"
            : AxisArgName == NAME_IS_CONCAT_DIM ? "concat_dim" : "<invalid>";
    OP_REQUIRES_OK(c, c->input(axis_attribute_name, &concat_dim_tensor));
    OP_REQUIRES(c, IsLegacyScalar(concat_dim_tensor->shape()),
                errors::InvalidArgument(
                    axis_attribute_name,
                    " tensor should be a scalar integer, but got shape ",
                    concat_dim_tensor->shape().DebugString()));
    const int32 concat_dim =
        internal::SubtleMustCopy(concat_dim_tensor->scalar<int32>()());
    // Instead of accessing values from context, we use input to Compute.
    const int N = values.size();
    const int input_dims = values[0].dims();
    const TensorShape& input_shape = values[0].shape();

    int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(c,
                (0 <= axis && axis < input_dims) ||
                    (allow_legacy_scalars() && concat_dim == 0),
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));
    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int64 output_concat_dim = 0;
    const bool input_is_scalar = IsLegacyScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto in = values[i];
      const bool in_is_scalar = IsLegacyScalar(in.shape());
      OP_REQUIRES(
          c, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i,
                "] = ", in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      // TODO(irving): Remove check once !allow_legacy_scalars().
      output_concat_dim += in.dims() > 0 ? in.dim_size(axis) : 1;
    }

    TensorShape output_shape(input_shape);
    // TODO(irving): Remove rank 0 case once !allow_legacy_scalars().
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(axis, output_concat_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() > 0) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
  }
};

// --------------------------------------------------------------------------
//                      Mkl Concat Op
// --------------------------------------------------------------------------

template <typename Device, typename T, AxisArgumentName AxisArgName>
class MklConcatOp : public OpKernel {
 private:
  TensorFormat data_format_;
  EigenConcatBaseOp<Device, T, AxisArgName> eigen_concat_op_;

 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit MklConcatOp(OpKernelConstruction* c)
      : OpKernel(c), eigen_concat_op_(c) {}

  void Compute(OpKernelContext* context) override {
    MklConcatOpContext mkl_context;

    // Get input tensors.
    OpInputList input_tensors;
    GetMklInputList(context, "values", &input_tensors);
    const int N = input_tensors.size();
    // Get MKL shapes.
    MklShapeList input_shapes(N);
    GetMklShapeList(context, "values", &input_shapes);

    // If this is Concat, then concat_dim is 0th input.
    // If this is ConcatV2, then axis is Nth input.
    const Tensor& concat_dim_tensor = AxisArgName == NAME_IS_CONCAT_DIM
                                          ? MklGetInput(context, 0)
                                          : MklGetInput(context, N);

    // Sanity checks
    OP_REQUIRES(
        context, IsLegacyScalar(concat_dim_tensor.shape()),
        errors::InvalidArgument(
            "Concat dim tensor should be a scalar integer, but got shape ",
            concat_dim_tensor.shape().DebugString()));
    int32 concat_dim =
        internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());

    MklShape& inpshape0 = input_shapes[0];

    // Check that all tensors are Mkl, if not we call Eigen version.
    bool invoke_eigen = false;
    bool is_concat_dim_channel = true;
    if (!AreAllMklTensors(input_shapes)) {
      invoke_eigen = true;
    }

    // Check that total number of dimensions is 4, if not call Eigen.
    if (!invoke_eigen) {
      for (auto& s : input_shapes) {
        if (s.GetDimension() != 4) {
          invoke_eigen = true;
          break;
        }
      }
    }

    // check that concat_dim is channel, if not call Eigen version.
    if (!invoke_eigen) {
      for (auto& s : input_shapes) {
        if (!s.IsMklChannelDim(concat_dim)) {
          invoke_eigen = true;
          is_concat_dim_channel = false;
          break;
        }
      }
    }

    if (invoke_eigen) {
      string msg = std::string("Invoking Eigen version of Concat. Reason:") +
                   (!is_concat_dim_channel
                        ? std::string("Concat dimension is not channel")
                        : std::string("Not all tensors are in Mkl layout"));
      VLOG(1) << "_MklConcatOp: " << msg;
      CallEigenVersion(context, input_tensors, input_shapes);
      return;
    }

    // For MKL format, the channel is dimension number 2.
    // So if we are concating over channel and _all_ inputs are in MKL
    // format, then we set concat_dim to 2.
    // Since we have reached till here, it means we are concating
    // over channel.
    concat_dim = MklDims::C;

    // One more sanity check: check that ranks of all tensors match
    // and that their shapes match except for concat_dim.
    int i = 0;
    for (auto& s : input_shapes) {
      size_t exp_dims = inpshape0.GetDimension();
      OP_REQUIRES(context, s.GetDimension() == exp_dims,
                  errors::InvalidArgument(
                      "_MklConcatOp : Ranks of all input tensors should match:"
                      " input dimensions = ",
                      s.GetDimension(), " vs. expected rank = ", exp_dims));

      for (int d = 0; d < exp_dims; ++d) {
        if (d == concat_dim) {
          continue;
        }

        size_t exp_size = inpshape0.GetSizes()[d];
        OP_REQUIRES(
            context, exp_size == s.GetSizes()[d],
            errors::InvalidArgument("_MklConcatOp : Dimensions of inputs"
                                    "should match: shape[0][",
                                    d, "]= ", exp_size, " vs. shape[", i, "][",
                                    d, "] = ", s.GetSizes()[d]));
      }
      ++i;
    }

    // Use input MKL layout instead of creating new layouts.
    int64 output_concat_dim_size = 0;
    for (auto& s : input_shapes) {
      output_concat_dim_size +=
          s.GetDimension() > 0 ? s.GetSizes()[concat_dim] : 1;
    }
    mkl_context.MklCreateInputLayouts(context, input_shapes);
    OP_REQUIRES_OK(context, context->status());

    CHECK_EQ(dnnConcatCreate_F32(&mkl_context.prim_concat, NULL, N,
                                 &mkl_context.lt_inputs[0]),
             E_SUCCESS);

    // Calculate output sizes and strides
    TensorFormat data_format;
    if (inpshape0.IsTensorInNHWCFormat()) {
      data_format = FORMAT_NHWC;
    } else {
      OP_REQUIRES(
          context, inpshape0.IsTensorInNCHWFormat(),
          errors::InvalidArgument(
              "_MklConcat only supports all inputs in NCHW or NHWC format "));
      data_format = FORMAT_NCHW;
    }

    // Since all tensors are in Mkl layout, we copy sizes from input tensor.
    mkl_context.out_sizes[MklDims::W] = inpshape0.GetSizes()[MklDims::W];
    mkl_context.out_sizes[MklDims::H] = inpshape0.GetSizes()[MklDims::H];
    mkl_context.out_sizes[MklDims::C] = output_concat_dim_size;
    mkl_context.out_sizes[MklDims::N] = inpshape0.GetSizes()[MklDims::N];
    GetStridesFromSizes(data_format, mkl_context.out_strides,
                        mkl_context.out_sizes);

    // Set output Mkl shape.
    int64 dim = 4;
    MklShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(true);
    mkl_output_mkl_shape.SetMklLayout(mkl_context.prim_concat, dnnResourceDst);
    mkl_output_mkl_shape.SetTfLayout(dim, mkl_context.out_sizes,
                                     mkl_context.out_strides);
    mkl_output_mkl_shape.SetTfDimOrder(dim, inpshape0.GetTfToMklDimMap());

    TensorShape mkl_output_tf_shape;
    mkl_output_tf_shape.AddDim(1);
    mkl_output_tf_shape.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mkl_output_mkl_shape.GetMklLayout())) /
        sizeof(T));

    Tensor* output = nullptr;
    AllocateOutputSetMklShape(context, 0, &output, mkl_output_tf_shape,
                              mkl_output_mkl_shape);

    // Set destination resource.
    mkl_context.concat_res[dnnResourceDst] =
        const_cast<void*>(static_cast<const void*>(output->flat<T>().data()));

    mkl_context.mkl_tmp_tensors.resize(N);
    mkl_context.MklPrepareConcatInputs(context, input_tensors);
    OP_REQUIRES_OK(context, context->status());

    // Execute primitive.
    CHECK_EQ(dnnExecute_F32(mkl_context.prim_concat, mkl_context.concat_res),
             E_SUCCESS);

    mkl_context.MklCleanup();
    OP_REQUIRES_OK(context, context->status());
  }

 private:
  typedef struct {
    TensorFormat data_format;
    size_t out_sizes[4];
    size_t out_strides[4];
    dnnPrimitive_t prim_concat;
    void* concat_res[dnnResourceNumber];
    std::vector<dnnLayout_t> lt_inputs;
    std::vector<Tensor> mkl_tmp_tensors;

    // Create MKL dnnLayout_t objects for tensors coming into the layer
    // We only support case where input tensors are all in Mkl layout.
    void MklCreateInputLayouts(OpKernelContext* context,
                               MklShapeList& input_shapes) {
      for (auto& is : input_shapes) {
        CHECK_EQ(is.IsMklTensor(), true);
        lt_inputs.push_back((dnnLayout_t)is.GetCurLayout());
      }
    }

    void MklPrepareConcatInputs(OpKernelContext* context,
                                OpInputList& input_tensors) {
      CHECK_EQ(lt_inputs.size(), mkl_tmp_tensors.size());

      for (int i = 0; i < lt_inputs.size(); ++i) {
        dnnPrimitive_t mkl_prim_convert_input;
        dnnLayout_t mkl_lt_internal_input;
        void* mkl_buf_convert_input = nullptr;

        CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                     &mkl_lt_internal_input, prim_concat,
                     (dnnResourceType_t)(dnnResourceMultipleSrc + i)),
                 E_SUCCESS);

        if (!dnnLayoutCompare_F32(lt_inputs[i], mkl_lt_internal_input)) {
          CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input,
                                           lt_inputs[i], mkl_lt_internal_input),
                   E_SUCCESS);

          AllocTmpBuffer(context, &mkl_tmp_tensors[i], mkl_lt_internal_input,
                         &mkl_buf_convert_input);

          CHECK_EQ(dnnConversionExecute_F32(
                       mkl_prim_convert_input,
                       const_cast<void*>(static_cast<const void*>(
                           input_tensors[i].flat<T>().data())),
                       mkl_buf_convert_input),
                   E_SUCCESS);

          concat_res[dnnResourceMultipleSrc + i] = mkl_buf_convert_input;
          CHECK_EQ(dnnDelete_F32(mkl_prim_convert_input), E_SUCCESS);
        } else {
          concat_res[dnnResourceMultipleSrc + i] = const_cast<void*>(
              static_cast<const void*>(input_tensors[i].flat<T>().data()));
        }

        CHECK_EQ(dnnLayoutDelete_F32(mkl_lt_internal_input), E_SUCCESS);
      }
    }

    void MklCleanup() {
      for (auto& lt : lt_inputs) {
        lt = nullptr;
      }
      CHECK_EQ(dnnDelete_F32(prim_concat), E_SUCCESS);
    }
  } MklConcatOpContext;

  void CallEigenVersion(OpKernelContext* context, const OpInputList& values,
                        const MklShapeList& input_shapes) {
    // Before calling Eigen version, we need to convert Mkl tensors to TF.
    // First check that the number of input tensors and the number of Mkl
    // shapes match.
    CHECK_EQ(values.size(), input_shapes.size());

    std::vector<Tensor> converted_values;
    for (int i = 0; i < input_shapes.size(); i++) {
      if (input_shapes[i].IsMklTensor()) {
        // If input tensor is Mkl, then do the conversion.
        Tensor tmp_tensor =
            ConvertMklToTF<T>(context, values[i], input_shapes[i]);
        converted_values.push_back(tmp_tensor);
      } else {
        // If input tensor is TF already, then we do not need any conversion.
        converted_values.push_back(values[i]);
      }
    }

    // Call Eigen concat.
    eigen_concat_op_.Compute(context, converted_values);

    // Set dummy Mkl tensor as output Mkl tensor for this op.
    MklShape mkl_tensor_mkl_shape;
    mkl_tensor_mkl_shape.SetMklTensor(false);
    mkl_tensor_mkl_shape.SetDimensions(4);
    mkl_tensor_mkl_shape.SetTfDimOrder(4);  // Dimensions
    Tensor* mkl_tensor = nullptr;
    TensorShape mkl_tensor_tf_shape;
    mkl_tensor_tf_shape.AddDim(
        SIZE_OF_MKL_SERIAL_DATA(mkl_tensor_mkl_shape.GetDimension()));
    int tf_output_index = 0;
    context->allocate_output(
        GetTensorMetaDataIndex(tf_output_index, context->num_outputs()),
        mkl_tensor_tf_shape, &mkl_tensor);
    mkl_tensor_mkl_shape.SerializeMklShape(
        mkl_tensor->flat<uint8>().data(),
        mkl_tensor->flat<uint8>().size() * sizeof(uint8));
  }
};

/* Use optimized concat for float type only */
#define REGISTER_MKL_CPU(type)                                              \
  REGISTER_KERNEL_BUILDER(Name("_MklConcat")                                \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .HostMemory("concat_dim")                     \
                              .Label(mkl_op_registry::kMklOpLabel),         \
                          MklConcatOp<CPUDevice, type, NAME_IS_CONCAT_DIM>) \
  REGISTER_KERNEL_BUILDER(Name("_MklConcatV2")                              \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<type>("T")                    \
                              .TypeConstraint<int32>("Tidx")                \
                              .HostMemory("axis")                           \
                              .Label(mkl_op_registry::kMklOpLabel),         \
                          MklConcatOp<CPUDevice, type, NAME_IS_AXIS>)

TF_CALL_float(REGISTER_MKL_CPU);

#undef REGISTER_CONCAT_MKL
}  // namespace tensorflow

#endif  // INTEL_MKL
