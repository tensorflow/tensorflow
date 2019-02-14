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
#include <unordered_map>
#include <vector>

#include "mkldnn.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/concat_lib_cpu.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::concat;
using mkldnn::stream;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

// List of TensorShape objects. Used in Concat/Split layers.
typedef std::vector<TensorShape> TensorShapeList;

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

  void Compute(OpKernelContext* c, const std::vector<Tensor>& values,
               const TensorShapeList& input_shapes) {
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
    const int input_dims = input_shapes[0].dims();
    const TensorShape& input_shape = input_shapes[0];

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
      const bool in_is_scalar = IsLegacyScalar(input_shapes[i]);
      OP_REQUIRES(
          c,
          (input_shapes[i].dims() == input_dims) ||
              (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", input_shapes[i].DebugString()));
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      output_concat_dim +=
          input_shapes[i].dims() > 0 ? input_shapes[i].dim_size(axis) : 1;
    }

    TensorShape output_shape(input_shape);
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
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      OpInputList input_tensors;
      GetMklInputList(context, "values", &input_tensors);
      const int N = input_tensors.size();

      // Get Tensor shapes.
      std::vector<MklDnnShape> mkl_input_shapes(N);
      GetMklShapeList(context, "values", &mkl_input_shapes);

      const Tensor& concat_dim_tensor = (AxisArgName == NAME_IS_CONCAT_DIM)
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

      // check that ranks of all tensors match
      // and that their shapes match except for concat_dim.
      int i = 0;
      bool invoke_eigen = false;
      bool are_all_mkl_inputs = true, are_all_tf_inputs = true;
      const TensorShape expected_shape = mkl_input_shapes[0].IsMklTensor()
                                             ? mkl_input_shapes[0].GetTfShape()
                                             : input_tensors[0].shape();
      size_t expected_dims = expected_shape.dims();

      if (concat_dim < 0) concat_dim = expected_dims + concat_dim;

      for (auto& s : mkl_input_shapes) {
        TensorShape s_shape =
            s.IsMklTensor() ? s.GetTfShape() : input_tensors[i].shape();
        size_t s_dims = s_shape.dims();

        OP_REQUIRES(
            context, s_dims == expected_dims,
            errors::InvalidArgument(
                "_MklConcatOp : Ranks of all input tensors should match:"
                " input dimensions = ",
                s_dims, " vs. expected rank = ", expected_dims));

        for (int d = 0; d < expected_dims; ++d) {
          if (d == concat_dim) continue;

          size_t expected_size = expected_shape.dim_size(d);
          size_t s_size = s_shape.dim_size(d);
          OP_REQUIRES(
              context, expected_size == s_size,
              errors::InvalidArgument("_MklConcatOp : Dimensions of inputs "
                                      "should match: shape[0][",
                                      d, "]= ", expected_size, " vs. shape[", i,
                                      "][", d, "] = ", s_size));
        }

        if (s.IsMklTensor())
          are_all_tf_inputs = false;
        else
          are_all_mkl_inputs = false;

        if (s_dims != 4) invoke_eigen = true;
        ++i;
      }

      // All inputs are not in one format (TF or MKL). This is mixed input case.
      // We can potentially optimize this case by converting all TF inputs
      // to Mkl format. But currently, we fall to Eigen for this case.
      // It may be possible to convert inputs that in TF format to Mkl
      // format and avoid calling eigen version.
      if (!are_all_tf_inputs && !are_all_mkl_inputs) invoke_eigen = true;

      OpInputList input_mins, input_maxes;
      if (std::is_same<T, qint8>::value || std::is_same<T, quint8>::value) {
        // MKL-DNN concat does not support input tensors that have different
        // ranges. Check if the ranges of the all input tensors are the same.
        // If not, forward it to Eigen implementation.

        OP_REQUIRES_OK(context, context->input_list("input_mins", &input_mins));
        OP_REQUIRES(context, (input_mins.size() == N),
                    errors::InvalidArgument(
                        "QuantizedConcatOp : Expected mins input list length ",
                        input_mins.size(), " to equal values length ", N));

        OP_REQUIRES_OK(context,
                       context->input_list("input_maxes", &input_maxes));
        OP_REQUIRES(context, (input_maxes.size() == N),
                    errors::InvalidArgument(
                        "QuantizedConcatOp : Expected maxes input list length ",
                        input_maxes.size(), " to equal values length ", N));
        float input_min = input_mins[0].flat<float>()(0);
        float input_max = input_maxes[0].flat<float>()(0);
        const float eps = 1.0e-6;
        for (int i = 1; i < N; ++i) {
          float min = input_mins[i].flat<float>()(0);
          float max = input_maxes[i].flat<float>()(0);

          if (fabs(input_min - min) > eps || fabs(input_max - max) > eps) {
            invoke_eigen = true;
            break;
          }
        }
      }

      // Call Eigen library
      if (invoke_eigen) {
        // MKL-DNN quantized concat does not support input tensors with
        // different ranges.
        // TODO (mabuzain): Add quantized version of CallEigen() to support
        // this case.
        OP_REQUIRES(
            context,
            (!std::is_same<T, qint8>::value && !std::is_same<T, quint8>::value),
            errors::Unimplemented("MKL DNN quantized concat does not "
                                  "support input tensors that have "
                                  "different ranges"));
        CallEigenVersion(context, input_tensors, mkl_input_shapes);
        return;
      }

      memory::dims dst_dims;

      if (are_all_mkl_inputs)
        dst_dims = TFShapeToMklDnnDims(mkl_input_shapes[0].GetTfShape());
      else
        // When all the inputs are in Tensorflow format, we don't know
        // what is the input data format. In that case, we just use
        // output format that is same as input formats.
        dst_dims = TFShapeToMklDnnDims(input_tensors[0].shape());

      std::vector<memory::primitive_desc> srcs_pd;
      std::vector<MklDnnData<T>> srcs(N, MklDnnData<T>(&cpu_engine));
      int64 dst_concat_dim_size = 0;

      bool isMklReorderNeeded = false;
      memory::format mkl_common_format = memory::format::any;
      if (are_all_mkl_inputs) {
        mkl_common_format =
            FindMklCommonFormat(mkl_input_shapes, concat_dim,
                                &isMklReorderNeeded, &dst_concat_dim_size);

        if (!isMklReorderNeeded) {
          // All MKL tensors have a same format. Reorder is not needed.
          for (int k = 0; k < N; k++) {
            if (input_tensors[k].NumElements() == 0) continue;

            auto src_md = mkl_input_shapes[k].GetMklLayout();
            srcs[k].SetUsrMem(src_md, &input_tensors[k]);
            auto src_mpd = srcs[k].GetUsrMemPrimDesc();
            srcs_pd.push_back(src_mpd);
          }
        } else {
          // MKL tensors have different formats.
          // Reorder them to most common format.
          for (int k = 0; k < N; k++) {
            if (input_tensors[k].NumElements() == 0) continue;

            auto src_md = mkl_input_shapes[k].GetMklLayout();
            srcs[k].SetUsrMem(src_md, &input_tensors[k]);

            if (src_md.data.format != mkl_common_format) {
              memory::dims src_dims(src_md.data.dims,
                                    &src_md.data.dims[src_md.data.ndims]);
              src_md =
                  memory::desc(src_dims, MklDnnType<T>(), mkl_common_format);
            }

            srcs_pd.push_back(memory::primitive_desc(src_md, cpu_engine));
          }
        }
      } else {  // All TF inputs
        for (int k = 0; k < N; k++) {
          if (input_tensors[k].NumElements() == 0) continue;

          memory::dims src_dims = TFShapeToMklDnnDims(input_tensors[k].shape());
          dst_concat_dim_size += src_dims[concat_dim];

          // It does not matter what data format to be used (NHWC versus NCHW).
          // We just need to ensure that output uses same data format as inputs.
          auto src_md =
              memory::desc(src_dims, MklDnnType<T>(), memory::format::nchw);

          srcs[k].SetUsrMem(src_md, &input_tensors[k]);
          auto src_mpd = srcs[k].GetUsrMemPrimDesc();
          srcs_pd.push_back(src_mpd);
        }
      }
      dst_dims[concat_dim] = dst_concat_dim_size;

      MklDnnData<T> dst(&cpu_engine);
      memory::desc dst_md({}, memory::data_undef, memory::format_undef);
      memory::dims dst_dims_in_nchw;
      if (are_all_mkl_inputs) {
        // Since we are passing a specific format for destination,
        // we need to have dst_dims in MklDnn order (NCHW).
        auto orig_tf_format = mkl_input_shapes[0].GetTfDataFormat();
        dst_dims_in_nchw = MklDnnDimsInNCHW(
            dst_dims, MklDnnDataFormatToTFDataFormat(orig_tf_format));
        // Set the output format same as the most common format of inputs
        // to avoid layout conversions.
        dst_md =
            memory::desc(dst_dims_in_nchw, MklDnnType<T>(), mkl_common_format);
      } else {
        // All inputs are TF tensors.
        // Set the output format same as input format (nchw).
        dst_md = memory::desc(dst_dims, MklDnnType<T>(), memory::format::nchw);
      }

      std::vector<primitive::at> inputs;
      if (isMklReorderNeeded) {
        for (int k = 0; k < input_tensors.size(); k++) {
          if (input_tensors[k].NumElements() > 0) {
            srcs[k].CheckReorderToOpMem(srcs_pd[k]);
          }
        }
      }
      for (int k = 0; k < input_tensors.size(); k++) {
        if (input_tensors[k].NumElements() > 0) {
          inputs.push_back(srcs[k].GetOpMem());
        }
      }

      // If all inputs are in MKL format, then meaning of concat_dim needs to
      // change. Value of concat_dim is tied to input Tensorflow data format
      // (NHWC or NCHW). MklDnn dimensions are in NCHW order. So if Tensorflow
      // tensors are in NCHW order, then concat_dim semantics is preserved.
      // But ifinput tensors are in NHWC order, then semantics need to change.
      // E.g., if we are concatinating over Channel (dimension 3 for NHWC),
      // then since MklDnn order is NCHW, concat_dim needs to be 1.
      if (are_all_mkl_inputs)
        concat_dim = mkl_input_shapes[0].TfDimIdx(concat_dim);

      auto concat_pd = concat::primitive_desc(concat_dim, srcs_pd);
      auto dst_pd = concat_pd.dst_primitive_desc();

      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      Tensor* dst_tensor = nullptr;
      if (are_all_mkl_inputs) {
        dnn_shape_dst.SetMklTensor(true);
        auto dst_pd = concat_pd.dst_primitive_desc();
        dnn_shape_dst.SetMklLayout(&dst_pd);
        dnn_shape_dst.SetElemType(MklDnnType<T>());
        dnn_shape_dst.SetTfLayout(dst_dims.size(), dst_dims_in_nchw,
                                  mkl_input_shapes[0].GetTfDataFormat());
        tf_shape_dst.AddDim((dst_pd.get_size() / sizeof(T)));
      } else {
        dnn_shape_dst.SetMklTensor(false);
        tf_shape_dst = MklDnnDimsToTFShape(dst_dims);
      }
      AllocateOutputSetMklShape(context, 0, &dst_tensor, tf_shape_dst,
                                dnn_shape_dst);
      CHECK_NOTNULL(dst_tensor);

      dst_md =
          dnn_shape_dst.IsMklTensor() ? dnn_shape_dst.GetMklLayout() : dst_md;
      dst.SetUsrMem(dst_md, dst_tensor);

      auto concat_op = concat(concat_pd, inputs, dst.GetOpMem());
      std::vector<primitive> net;
      net.push_back(concat_op);
      stream(stream::kind::eager).submit(net).wait();

      // For quantized concat, min and max outputs are also computed.
      if (std::is_same<T, qint8>::value || std::is_same<T, quint8>::value) {
        Tensor* output_min = nullptr;
        Tensor* output_max = nullptr;
        MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
        output_min_mkl_shape.SetMklTensor(false);
        output_max_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(context, 1, &output_min, {},
                                  output_min_mkl_shape);
        AllocateOutputSetMklShape(context, 2, &output_max, {},
                                  output_max_mkl_shape);
        // All input tensors should have the same range, just use the
        // first one
        output_min->flat<float>()(0) = input_mins[0].flat<float>()(0);
        output_max->flat<float>()(0) = input_maxes[0].flat<float>()(0);
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void CallEigenVersion(OpKernelContext* context, const OpInputList& values,
                        const MklDnnShapeList& mkl_input_shapes) {
    CHECK_EQ(values.size(), mkl_input_shapes.size());

    std::vector<Tensor> converted_values;
    TensorShapeList tf_input_shapes;
    for (int i = 0; i < mkl_input_shapes.size(); i++) {
      if (mkl_input_shapes[i].IsMklTensor()) {
        // do conversion from MKL to TF
        Tensor tmp_tensor =
            ConvertMklToTF<T>(context, values[i], mkl_input_shapes[i]);
        converted_values.push_back(tmp_tensor);
        tf_input_shapes.push_back(mkl_input_shapes[i].GetTfShape());
      } else {
        // no conversion since it is TF tensor already
        converted_values.push_back(values[i]);
        tf_input_shapes.push_back(values[i].shape());
      }
    }

    // Call Eigen concat.
    eigen_concat_op_.Compute(context, converted_values, tf_input_shapes);

    // Set output Mkl tensor for this op.
    MklDnnShape dnn_shape_output;
    dnn_shape_output.SetMklTensor(false);
    dnn_shape_output.SetDimensions(4);
    Tensor* output_tensor = nullptr;
    TensorShape tf_shape_output;
    tf_shape_output.AddDim(dnn_shape_output.GetSerializeBufferSize());
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       GetTensorMetaDataIndex(0, context->num_outputs()),
                       tf_shape_output, &output_tensor));
    dnn_shape_output.SerializeMklDnnShape(
        output_tensor->flat<uint8>().data(),
        output_tensor->flat<uint8>().size() * sizeof(uint8));
  }

  // This method finds the most commom format accross all MKL inputs
  // Inputs:
  //   1. input_shapes: shapes of input (MKL) tensors.
  //   2. concat_dim: concat dimension.
  // Outputs:
  //   1. is_reorder_needed is set to true if inputs have difference formats
  //      It is set to false otherwise.
  //   2. concat_dim_size is the size of concat_dim.
  // Return:
  //   return the common MKL format.
  memory::format FindMklCommonFormat(const MklDnnShapeList& input_shapes,
                                     int concat_dim, bool* is_reorder_needed,
                                     int64* concat_dim_size) {
    *is_reorder_needed = false;
    *concat_dim_size = 0;
    std::unordered_map<int, int> occurrence_map;
    if (input_shapes.size() == 0) return memory::format::any;

    // Compute ocurrences of each format of all inputs.
    for (int k = 0; k < input_shapes.size(); k++) {
      auto src_dims = TFShapeToMklDnnDims(input_shapes[k].GetTfShape());
      *concat_dim_size += src_dims[concat_dim];
      int fmt = static_cast<int>(input_shapes[k].GetMklLayout().data.format);
      occurrence_map[fmt] += 1;
    }

    if (occurrence_map.size() == 1) {
      // this means that all inputs have a same format
      // return it with is_reorder_needed set false.
      return static_cast<memory::format>(
          input_shapes[0].GetMklLayout().data.format);
    }

    // Input tensors have different formats. Thus, reorder is needed.
    // We pick up the most common format to minimize the total
    // number of input reorder.
    memory::format commonest_format = memory::format::any;
    int max_occurrence = 0;
    *is_reorder_needed = true;
    for (auto item : occurrence_map) {
      if (item.second > max_occurrence) {
        commonest_format = static_cast<memory::format>(item.first);
        max_occurrence = item.second;
      }
    }
    return commonest_format;
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

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .HostMemory("axis")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklConcatOp<CPUDevice, quint8, NAME_IS_AXIS>)

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .HostMemory("axis")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklConcatOp<CPUDevice, qint8, NAME_IS_AXIS>)

#undef REGISTER_CONCAT_MKL
}  // namespace tensorflow

#endif  // INTEL_MKL
