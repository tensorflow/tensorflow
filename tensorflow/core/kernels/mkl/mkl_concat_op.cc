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
#define EIGEN_USE_THREADS

#include <limits>
#include <unordered_map>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "dnnl.hpp"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/concat_lib_cpu.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/mkl_util.h"
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::concat;
using dnnl::stream;

namespace tensorflow {
#ifndef ENABLE_ONEDNN_V3
#define CONCAT_PRIM_DESC(eng, concat_dims, src_md, dst_md_ptr) \
  concat::primitive_desc(*dst_md_ptr, concat_dims, src_md, eng)
#define CONCAT_PRIM_DESC_USING_SRC(eng, concat_dims, src_md) \
  concat::primitive_desc(concat_dims, src_md, eng)
#define GET_MEMORY_DESC(md) md.data
#define SET_MKL_LAYOUT(md) SetMklLayout(&md)
#else
#define CONCAT_PRIM_DESC(eng, concat_dims, src_md, dst_md_ptr) \
  concat::primitive_desc(eng, *dst_md_ptr, concat_dims, src_md)
#define CONCAT_PRIM_DESC_USING_SRC(eng, concat_dims, src_md) \
  concat::primitive_desc(eng, concat_dims, src_md)
#define GET_MEMORY_DESC(md) md
#define SET_MKL_LAYOUT(md) SetMklLayout(md)
#endif  // !ENABLE_ONEDNN_V3
typedef Eigen::ThreadPoolDevice CPUDevice;

// List of TensorShape objects. Used in Concat/Split layers.
typedef std::vector<TensorShape> TensorShapeList;

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

// TODO(intel-tf) Check if we can reuse existing EigenConcatOp using Mutable
// reference inputs.
// --------------------------------------------------------------------------
//                      Eigen Concat Op
// --------------------------------------------------------------------------
namespace {
template <typename T>
struct RequantizeCopier {
  RequantizeCopier(
      const std::vector<std::pair<float, float>>* input_min_and_max,
      float output_min, float output_max)
      : output_min(output_min), output_max(output_max) {
    DCHECK(input_min_and_max);
    this->input_min_and_max = input_min_and_max;
  }

  inline void Copy(T* dst, const T* src, int input_index, size_t n) {
    const float input_min = (*input_min_and_max)[input_index].first;
    const float input_max = (*input_min_and_max)[input_index].second;
    if (input_min == output_min && input_max == output_max) {
      DCHECK(DataTypeCanUseMemcpy(DataTypeToEnum<T>::v()));
      memcpy(dst, src, n * sizeof(T));
    } else {
      Eigen::array<Eigen::DenseIndex, 1> dims;
      dims[0] = n;
      typename TTypes<T, 1>::UnalignedConstTensor input_array(src, dims);
      typename TTypes<T, 1>::UnalignedTensor output_array(dst, dims);

      QuantizedToFloatStruct<T> q2f(input_min, input_max);
      auto input_float = DEQUANTIZE_WITH_EIGEN(input_array, q2f);
      FloatToQuantizedStruct<T> f2q(output_min, output_max);
      // RequantizeCopier::Copy is called from within a shard of computation, so
      // don't use the threadpool device here, simply assign with default CPU
      // device.
      output_array = QUANTIZE_WITH_EIGEN(input_float, f2q, T);
    }
  }

  float output_min;
  float output_max;
  const std::vector<std::pair<float, float>>* input_min_and_max;
};
}  // namespace

template <typename Device, typename T, AxisArgumentName AxisArgName>
class EigenConcatBaseOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit EigenConcatBaseOp(OpKernelConstruction* c) : OpKernel(c) {}

  void CalculateInputAndOutputRange(
      const OpInputList& input_mins, const OpInputList& input_maxes,
      const size_t N,
      std::vector<std::pair<float, float>>* input_mins_and_maxes,
      float* output_min, float* output_max) {
    input_mins_and_maxes->reserve(N);
    float overall_min = std::numeric_limits<float>::max();
    float overall_max = std::numeric_limits<float>::lowest();
    for (int i = 0; i < N; ++i) {
      const float input_min = input_mins[i].scalar<float>()();
      const float input_max = input_maxes[i].scalar<float>()();
      input_mins_and_maxes->emplace_back(input_min, input_max);
      overall_min = std::min(overall_min, input_min);
      overall_max = std::max(overall_max, input_max);
    }
    if (std::numeric_limits<T>::is_signed) {
      // For signed, we want a symmetrical distribution including zero for the
      // output, so pick a range that meets that need.
      const float largest_value =
          std::max(std::abs(overall_min), std::abs(overall_max));
      *output_min = -largest_value;
      *output_max = largest_value;
    } else {
      // For MKL quantization, we only support scaled mode, so the range is
      // [0, m] for unsigned data where m is the range maximum
      *output_min = 0.0f;
      *output_max = overall_max;
    }
  }

  // Although, we modify Compute for this call to accept one extra param,
  // we need to have empty Compute because Compute is pure virtual function.
  void Compute(OpKernelContext* c) {}

  void Compute(OpKernelContext* c, const std::vector<Tensor>& values,
               const TensorShapeList& input_shapes,
               const OpInputList& input_mins, const OpInputList& input_maxes,
               bool quantized_input) {
    const Tensor* concat_dim_tensor;
    const char* axis_attribute_name = AxisArgName == NAME_IS_AXIS ? "axis"
                                      : AxisArgName == NAME_IS_CONCAT_DIM
                                          ? "concat_dim"
                                          : "<invalid>";
    OP_REQUIRES_OK(c, c->input(axis_attribute_name, &concat_dim_tensor));
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(concat_dim_tensor->shape()),
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

    int32 axis = (concat_dim < 0) ? (concat_dim + input_dims) : concat_dim;
    OP_REQUIRES(
        c, (0 <= axis && axis < input_dims),
        errors::InvalidArgument(
            "ConcatOp : Expected concatenating dimensions in the range [",
            -input_dims, ", ", input_dims, "), but got ", concat_dim));

    float output_min = std::numeric_limits<float>::max();
    float output_max = std::numeric_limits<float>::lowest();
    std::vector<std::pair<float, float>> input_mins_and_maxes;
    if (quantized_input) {
      CalculateInputAndOutputRange(input_mins, input_maxes, N,
                                   &input_mins_and_maxes, &output_min,
                                   &output_max);
    }
    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x_0, x_1,...,x_n-1, y_0, y_1,...,y_m-1}, where the
    // concat is along the dimension indicated with size y_0, we flatten it
    // to {x, y}, where y = Prod_i(y_i) and x = ((n > 0) ? Prod_i(x_i) : 1).
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int64 output_concat_dim = 0;
    const bool input_is_scalar = TensorShapeUtils::IsScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto in = values[i];
      const bool in_is_scalar = TensorShapeUtils::IsScalar(input_shapes[i]);
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
      if (!quantized_input) {
        ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
      } else {
        ConcatCPUImpl<T>(
            c->device(), inputs_flat, sizeof(T) /* cost_per_unit */,
            RequantizeCopier<T>(&input_mins_and_maxes, output_min, output_max),
            &output_flat);
      }
    }

    if (quantized_input) {
      Tensor* output_min_tensor = nullptr;
      OP_REQUIRES_OK(c, c->allocate_output(1, {}, &output_min_tensor));
      output_min_tensor->scalar<float>()() = output_min;

      Tensor* output_max_tensor = nullptr;
      OP_REQUIRES_OK(c, c->allocate_output(2, {}, &output_max_tensor));
      output_max_tensor->scalar<float>()() = output_max;
    }
  }
};
// --------------------------------------------------------------------------
//                      Mkl Concat Op
// --------------------------------------------------------------------------
// This structure aggregates multiple inputs to MklConcat* methods.
struct MklConcatFwdParams {
  std::vector<memory::dims> src_dims;
  memory::dims dst_dims;
  int num_inputs;
  int concat_dims;
  memory::format_tag mkl_common_format;

  MklConcatFwdParams(std::vector<memory::dims>& src_dims_pt,
                     memory::dims dst_dims, int num_inputs, int concat_dims,
                     memory::format_tag mkl_common_format)
      : dst_dims(dst_dims),
        num_inputs(num_inputs),
        concat_dims(concat_dims),
        mkl_common_format(mkl_common_format) {
    for (int k = 0; k < num_inputs; ++k) {
      src_dims.push_back(src_dims_pt[k]);
    }
  }
};

// TODO(intel-tf): The template type "T" is currently used to match the
// templatized class MklPrimitiveFactory (tensorflow/core/util/mkl_util.h).
// In the future, with the removal of "T" from MklPrimitiveFactory, this class
// needs to drop "T".
template <typename T>
class MklConcatFwdPrimitive : public MklPrimitive {
 public:
  explicit MklConcatFwdPrimitive(const MklConcatFwdParams& concat_fwd_dims,
                                 const std::vector<memory::desc>& srcs_md)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create concat primitive
    Setup(concat_fwd_dims, srcs_md);
  }

  ~MklConcatFwdPrimitive() {}

  // Concat forward execute
  //   src_data:    input data buffer of src
  //   dst_data:    output data buffer of dst
  void Execute(const std::vector<dnnl::memory>& in_data,
               const dnnl::memory& dst_data,
               const MklConcatFwdParams& concat_fwd_dims,
               std::shared_ptr<stream> fwd_stream) {
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    mutex_lock lock(primitive_execution_mu_);
#endif
    DCHECK_EQ(in_data.size(), context_.data_mem.size());
    for (size_t i = 0; i < concat_fwd_dims.num_inputs; i++) {
#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
      context_.data_mem_shdptr[i]->set_data_handle(
          static_cast<void*>(in_data[i].get_data_handle()), *fwd_stream);
    }
    context_.dst_mem->set_data_handle(
        static_cast<void*>(dst_data.get_data_handle()), *fwd_stream);
#else
      context_.data_mem_shdptr[i]->set_data_handle(
          static_cast<void*>(in_data[i].get_data_handle()));
    }
    context_.dst_mem->set_data_handle(
        static_cast<void*>(dst_data.get_data_handle()));
#endif  // !ENABLE_ONEDNN_OPENMP && !ENABLE_ONEDNN_V3

    for (size_t i = 0; i < concat_fwd_dims.num_inputs; i++) {
      context_.data_mem[i] = *context_.data_mem_shdptr[i];
    }

    execute_primitives(context_.fwd_primitives, fwd_stream,
                       context_.fwd_primitives_args);

    // After exec, set data handle back
    context_.dst_mem->set_data_handle(DummyData);
    for (int k = 0; k < concat_fwd_dims.num_inputs; k++) {
      context_.data_mem_shdptr[k]->set_data_handle(DummyData);
    }

    for (size_t i = 0; i < concat_fwd_dims.num_inputs; i++) {
      context_.data_mem[i] = *context_.data_mem_shdptr[i];
    }
  }

 private:
  // Primitive reuse context for concat Fwd op
  struct ConcatFwdContext {
    // oneDNN memory
    std::vector<dnnl::memory> data_mem;
    std::vector<std::shared_ptr<dnnl::memory>> data_mem_shdptr;
    std::shared_ptr<dnnl::memory> dst_mem;

    // Memory descriptor
    std::vector<dnnl::memory::desc> src_md;
    std::shared_ptr<dnnl::memory::desc> dst_md;

    // Concat primitive descriptor
    std::shared_ptr<dnnl::concat::primitive_desc> fwd_pd;
    std::shared_ptr<dnnl::primitive> concat_fwd;

    std::vector<dnnl::primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> fwd_primitives_args;

    ConcatFwdContext()
        : dst_mem(nullptr), fwd_pd(nullptr), concat_fwd(nullptr) {}
  };

  // Creates the src and dst memory descriptor for mkl concat
  // and also creates the concat primitive and primitive descriptor
  void Setup(const MklConcatFwdParams& concat_fwd_dims,
             const std::vector<memory::desc>& srcs_md) {
    // Create memory descriptors for concat with specified srcs format
    for (size_t i = 0; i < concat_fwd_dims.num_inputs; i++) {
      dnnl::memory::desc source_md((memory::desc(GET_MEMORY_DESC(srcs_md[i]))));
      context_.src_md.push_back(source_md);
      std::shared_ptr<dnnl::memory> src_mem(
          new dnnl::memory(source_md, cpu_engine_, DummyData));
      context_.data_mem_shdptr.push_back(src_mem);
      context_.data_mem.push_back(*context_.data_mem_shdptr[i]);
    }
    // Store the expected memory format
    context_.dst_md.reset(new memory::desc({concat_fwd_dims.dst_dims},
                                           MklDnnType<T>(),
                                           concat_fwd_dims.mkl_common_format));
    // Create a concat primitive descriptor
    context_.fwd_pd.reset(
        new CONCAT_PRIM_DESC(cpu_engine_, concat_fwd_dims.concat_dims,
                             context_.src_md, context_.dst_md));

    // Create memory primitive based on dummy data
    context_.dst_mem.reset(
        new memory(*context_.dst_md, cpu_engine_, DummyData));

    context_.concat_fwd.reset(new concat(*context_.fwd_pd));
    std::unordered_map<int, memory> net_args = {
        {DNNL_ARG_DST, *context_.dst_mem}};
    for (int i = 0; i < concat_fwd_dims.num_inputs; ++i) {
      net_args.insert({DNNL_ARG_MULTIPLE_SRC + i, context_.data_mem[i]});
    }

    context_.fwd_primitives_args.push_back(net_args);
    context_.fwd_primitives.push_back(*context_.concat_fwd);
  }

  struct ConcatFwdContext context_;

#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  mutex primitive_execution_mu_;
#endif
};

// Class to create/cache the mkl concat primitives based on the
// input and output parameters
template <typename T>
class MklConcatFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklConcatFwdPrimitive<T>* Get(
      const MklConcatFwdParams& concat_fwd_dims,
      const std::vector<memory::desc>& srcs_md, bool do_not_cache) {
    MklConcatFwdPrimitive<T>* concat_fwd = nullptr;

    if (do_not_cache) {
      // Always create new primitive
      concat_fwd = new MklConcatFwdPrimitive<T>(concat_fwd_dims, srcs_md);
    } else {
      // Try to find a suitable one in pool
      concat_fwd = dynamic_cast<MklConcatFwdPrimitive<T>*>(
          MklConcatFwdPrimitiveFactory<T>::GetInstance().GetConcatFwd(
              concat_fwd_dims));
      if (concat_fwd == nullptr) {
        concat_fwd = new MklConcatFwdPrimitive<T>(concat_fwd_dims, srcs_md);
        MklConcatFwdPrimitiveFactory<T>::GetInstance().SetConcatFwd(
            concat_fwd_dims, concat_fwd);
      }
    }

    return concat_fwd;
  }

 private:
  MklConcatFwdPrimitiveFactory() {}
  ~MklConcatFwdPrimitiveFactory() {}

  static MklConcatFwdPrimitiveFactory& GetInstance() {
    static MklConcatFwdPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklConcatFwdParams& concat_fwd_dims) {
    string prefix = "concat_fwd_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    for (int k = 0; k < concat_fwd_dims.num_inputs; k++) {
      key_creator.AddAsKey(concat_fwd_dims.src_dims[k]);
    }
    key_creator.AddAsKey(concat_fwd_dims.concat_dims);
    return key_creator.GetKey();
  }

  MklPrimitive* GetConcatFwd(const MklConcatFwdParams& concat_fwd_dims) {
    string key = CreateKey(concat_fwd_dims);
    return this->GetOp(key);
  }

  void SetConcatFwd(const MklConcatFwdParams& concat_fwd_dims,
                    MklPrimitive* op) {
    string key = CreateKey(concat_fwd_dims);
    this->SetOp(key, op);
  }
};

template <typename Device, typename T, AxisArgumentName AxisArgName,
          bool native_format = false>
class MklConcatOp : public OpKernel {
 private:
  TensorFormat data_format_;
  EigenConcatBaseOp<Device, T, AxisArgName> eigen_concat_op_;

 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit MklConcatOp(OpKernelConstruction* c)
      : OpKernel(c),
        data_format_(TensorFormat::FORMAT_NCHW),
        eigen_concat_op_(c) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::kind::cpu, 0);
      OpInputList input_tensors(context, 0, 0);
      GetMklInputList(context, "values", &input_tensors);
      const int N = input_tensors.size();
      // Get Tensor shapes.
      std::vector<MklDnnShape> mkl_input_shapes(N);
      GetMklShapeList(context, "values", &mkl_input_shapes, native_format);

      const Tensor& concat_dim_tensor = (AxisArgName == NAME_IS_CONCAT_DIM)
                                            ? MklGetInput(context, 0)
                                            : MklGetInput(context, N);
      // Sanity checks
      OP_REQUIRES(
          context, TensorShapeUtils::IsScalar(concat_dim_tensor.shape()),
          errors::InvalidArgument(
              "Concat dim tensor should be a scalar integer, but got shape ",
              concat_dim_tensor.shape().DebugString()));
      int32 concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());

      // check that ranks of all tensors match
      // and that their shapes match except for concat_dim.
      int i = 0;
      int num_of_empty_inputs = 0;
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

        if (s_dims != 4 && s_dims != 2) invoke_eigen = true;

        if (input_tensors[i].NumElements() == 0) num_of_empty_inputs++;

        ++i;
      }

      if (num_of_empty_inputs == i) invoke_eigen = true;

      // All inputs are not in one format (TF or MKL). This is mixed input case.
      // We can potentially optimize this case by converting all TF inputs
      // to Mkl format. But currently, we fall to Eigen for this case.
      // It may be possible to convert inputs that in TF format to Mkl
      // format and avoid calling eigen version.
      if (!are_all_tf_inputs && !are_all_mkl_inputs) invoke_eigen = true;

      // Temporally call Eigen if number of input dimensions is 2.
      // That is due to an incorrect output results in DNNL 1.2 path.
      if (expected_dims == 2) invoke_eigen = true;

      OpInputList input_mins(context, 0, 0);
      OpInputList input_maxes(context, 0, 0);
      bool quantized_input =
          std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;
      if (quantized_input) {
        // oneDNN concat does not support input tensors that have different
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

        for (int i = 0; i < N; i++) {
          OP_REQUIRES(context,
                      TensorShapeUtils::IsScalar(input_mins[i].shape()),
                      errors::InvalidArgument("`input_mins[", i,
                                              "]` must be rank 0 but is rank ",
                                              input_mins[i].dims()));
          OP_REQUIRES(context,
                      TensorShapeUtils::IsScalar(input_maxes[i].shape()),
                      errors::InvalidArgument("`input_maxes[", i,
                                              "]` must be rank 0 but is rank ",
                                              input_maxes[i].dims()));
        }
        float input_min = input_mins[0].scalar<float>()();
        float input_max = input_maxes[0].scalar<float>()();
        const float eps = 1.0e-6;
        for (int i = 1; i < N; ++i) {
          float min = input_mins[i].scalar<float>()();
          float max = input_maxes[i].scalar<float>()();

          if (fabs(input_min - min) > eps || fabs(input_max - max) > eps) {
            invoke_eigen = true;
            break;
          }
        }
      }

      // Call Eigen library
      if (invoke_eigen) {
        CallEigenVersion(context, input_tensors, input_mins, input_maxes,
                         mkl_input_shapes, quantized_input);
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

      std::vector<memory::desc> srcs_pd;
      std::vector<MklDnnData<T>> srcs(N, MklDnnData<T>(&cpu_engine));
      int64 dst_concat_dim_size = 0;

      bool isMklReorderNeeded = false;
      memory::format_tag mkl_common_format = memory::format_tag::any;
      std::vector<memory> inputs;
      std::vector<memory::dims> src_dims_pt;
      std::vector<dnnl::memory> srcs_mem;
      std::vector<memory::desc> srcs_md;

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
            auto src_mpd = srcs[k].GetUsrMemDesc();
            srcs_pd.push_back(src_mpd);
            inputs.push_back(srcs[k].GetOpMem());
          }
        } else {
          // MKL tensors have different formats.
          // Reorder them to most common format.
          for (int k = 0; k < N; k++) {
            if (input_tensors[k].NumElements() == 0) continue;
            auto src_md = mkl_input_shapes[k].GetMklLayout();
            srcs[k].SetUsrMem(src_md, &input_tensors[k]);
            auto src_tf_fmt = MklTensorFormatToMklDnnDataFormat(
                mkl_input_shapes[k].GetTfDataFormat());
            if (src_tf_fmt != mkl_common_format) {
#ifndef ENABLE_ONEDNN_V3
              memory::dims src_dims(src_md.data.dims,
                                    &src_md.data.dims[src_md.data.ndims]);
#else
              memory::dims src_dims;
              if (src_md.get_ndims() == 2)
                src_dims = {src_md.get_dims()[0], src_md.get_dims()[1]};
              else if (src_md.get_ndims() == 4)
                src_dims = {src_md.get_dims()[0], src_md.get_dims()[1],
                            src_md.get_dims()[2], src_md.get_dims()[3]};
#endif  // !ENABLE_ONEDNN_V3
              src_md =
                  memory::desc(src_dims, MklDnnType<T>(), mkl_common_format);
            }
            srcs_pd.push_back(memory::desc(src_md));
          }
        }
      } else {  // All TF inputs
        for (int k = 0; k < N; k++) {
          if (input_tensors[k].NumElements() == 0) continue;
          TensorShape s_shape = input_tensors[k].shape();
          memory::dims src_dims = TFShapeToMklDnnDims(s_shape);
          dst_concat_dim_size += src_dims[concat_dim];
          size_t s_dims = s_shape.dims();

          // It does not matter what data format to be used (NHWC versus NCHW).
          // We just need to ensure that output uses same data format as inputs.
          if (s_dims == 4)
            mkl_common_format = memory::format_tag::nchw;
          else if (s_dims == 2)
            mkl_common_format = memory::format_tag::nc;

          auto src_md =
              memory::desc(src_dims, MklDnnType<T>(), mkl_common_format);

          srcs[k].SetUsrMem(src_md, &input_tensors[k]);
          auto src_mpd = srcs[k].GetUsrMemDesc();
          srcs_pd.push_back(src_mpd);
          inputs.push_back(srcs[k].GetOpMem());
          src_dims_pt.push_back(src_dims);
          srcs_md.push_back(src_md);
          srcs_mem.push_back(srcs[k].GetOpMem());
        }
      }
      dst_dims[concat_dim] = dst_concat_dim_size;

      MklDnnData<T> dst(&cpu_engine);
      memory::desc dst_md({}, memory::data_type::undef,
                          memory::format_tag::undef);
      memory::dims dst_dims_in_nchw;
      if (are_all_mkl_inputs) {
        // Since we are passing a specific format for destination,
        // we need to have dst_dims in MklDnn order (NCHW).
        auto orig_tf_format = mkl_input_shapes[0].GetTfDataFormat();
        if (dst_dims.size() == 4) {
          dst_dims_in_nchw = MklDnnDimsInNCHW(
              dst_dims, MklDnnDataFormatToTFDataFormat(orig_tf_format));
          // Set the output format same as the most common format of inputs
          // to avoid layout conversions.
          // DNN 1.0: internal format is always blocked;
          //          format_tag does not have "blocked" field.
          VLOG(1) << "mkl_common_format == memory::format_tag::blocked";
          dst_md = MklDnnData<T>::CreateBlockedMemDesc(
              dst_dims_in_nchw, CalculateTFStrides(dst_dims_in_nchw));
        } else if (dst_dims.size() == 2 &&
                   mkl_common_format == memory::format_tag::nc) {
          // When memory::format_tag::nc, dst_dims are already in oneDNN order
          dst_md = memory::desc(dst_dims, MklDnnType<T>(), mkl_common_format);
        } else {
          TF_CHECK_OK(Status(absl::StatusCode::kFailedPrecondition,
                             "Unsupported tensor dimension or"
                             "oneDNN memory format"));
        }
      } else {
        // All inputs are TF tensors.
        // Set the output format same as input format (nchw/nc).
        dst_md = memory::desc(dst_dims, MklDnnType<T>(), mkl_common_format);
      }

      if (isMklReorderNeeded) {
        for (int k = 0; k < input_tensors.size(); k++) {
          if (input_tensors[k].NumElements() > 0) {
            srcs[k].CheckReorderToOpMem(srcs_pd[k], cpu_engine, context);
            inputs.push_back(srcs[k].GetOpMem());
          }
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
      // Create the oneDNN wrapper over Eigen threadpool and set max threads
      // in oneDNN.
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(context);
      tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                     ThreadPoolUseCallerThread());
      if (!inputs.empty()) {
        if (are_all_mkl_inputs) {
          auto concat_pd =
              CONCAT_PRIM_DESC_USING_SRC(cpu_engine, concat_dim, srcs_pd);
          auto dst_pd = concat_pd.dst_desc();

          MklDnnShape dnn_shape_dst;
          TensorShape tf_shape_dst;
          Tensor* dst_tensor = nullptr;
          dnn_shape_dst.SetMklTensor(true);
          dnn_shape_dst.SET_MKL_LAYOUT(dst_pd);
          dnn_shape_dst.SetElemType(MklDnnType<T>());
          dnn_shape_dst.SetTfLayout(dst_dims.size(), dst_dims_in_nchw,
                                    mkl_input_shapes[0].GetTfDataFormat());
          tf_shape_dst.AddDim((dst_pd.get_size() / sizeof(T)));
          AllocateOutputSetMklShape(context, 0, &dst_tensor, tf_shape_dst,
                                    dnn_shape_dst);
          DCHECK(dst_tensor != nullptr) << "Output tensor pointer is NULL";

          std::shared_ptr<stream> fwd_cpu_stream;

          fwd_cpu_stream.reset(CreateStream(&eigen_tp, cpu_engine));

          if (dnn_shape_dst.IsMklTensor())
            dst_md = dnn_shape_dst.GetMklLayout();
          dst.SetUsrMem(dst_md, dst_tensor);
          dst.SetUsrMemDataHandle(dst_tensor, fwd_cpu_stream);

          auto concat_op = concat(concat_pd);
          std::unordered_map<int, memory> net_args = {
              {DNNL_ARG_DST, dst.GetOpMem()}};
          for (int i = 0; i < inputs.size(); ++i) {
            net_args.insert({DNNL_ARG_MULTIPLE_SRC + i, inputs[i]});
          }
          concat_op.execute(*fwd_cpu_stream, net_args);
        } else {
          MklConcatFwdPrimitive<T>* concat_fwd = nullptr;

          MklConcatFwdParams concat_fwd_dims(src_dims_pt, dst_dims,
                                             (N - num_of_empty_inputs),
                                             concat_dim, mkl_common_format);
          // Get a concat fwd from primitive pool
          concat_fwd =
              MklConcatFwdPrimitiveFactory<T>::Get(concat_fwd_dims, srcs_md, 0);

          // Allocate output tensor.
          MklDnnShape dnn_shape_dst;
          TensorShape tf_shape_dst;
          Tensor* dst_tensor = nullptr;
          dnn_shape_dst.SetMklTensor(false);
          tf_shape_dst = MklDnnDimsToTFShape(dst_dims);
          AllocateOutputSetMklShape(context, 0, &dst_tensor, tf_shape_dst,
                                    dnn_shape_dst, native_format);
          DCHECK(dst_tensor != nullptr) << "Output tensor pointer is NULL";

          dst_md = dnn_shape_dst.IsMklTensor() ? dnn_shape_dst.GetMklLayout()
                                               : dst_md;
          std::shared_ptr<stream> fwd_cpu_stream;

          fwd_cpu_stream.reset(
              CreateStream(&eigen_tp, concat_fwd->GetEngine()));
          dst.SetUsrMem(dst_md, dst_tensor);
          dst.SetUsrMemDataHandle(dst_tensor, fwd_cpu_stream);
          // Execute concat
          concat_fwd->Execute(srcs_mem, dst.GetOpMem(), concat_fwd_dims,
                              fwd_cpu_stream);
        }

        // For quantized concat, min and max outputs are also computed.
        if (quantized_input) {
          Tensor* output_min = nullptr;
          Tensor* output_max = nullptr;
          MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
          output_min_mkl_shape.SetMklTensor(false);
          output_max_mkl_shape.SetMklTensor(false);
          AllocateOutputSetMklShape(context, 1, &output_min, {},
                                    output_min_mkl_shape, native_format);
          AllocateOutputSetMklShape(context, 2, &output_max, {},
                                    output_max_mkl_shape, native_format);
          // All input tensors should have the same range, just use the
          // first one
          output_min->scalar<float>()() = input_mins[0].scalar<float>()();
          output_max->scalar<float>()() = input_maxes[0].scalar<float>()();
        }
      } else {
        MklDnnShape dnn_shape_dst;
        TensorShape tf_shape_dst;
        Tensor* dst_tensor = nullptr;
        dnn_shape_dst.SetMklTensor(false);
        tf_shape_dst = MklDnnDimsToTFShape(dst_dims);

        AllocateOutputSetMklShape(context, 0, &dst_tensor, tf_shape_dst,
                                  dnn_shape_dst, native_format);
        DCHECK(dst_tensor != nullptr) << "Output tensor pointer is NULL";
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void CallEigenVersion(OpKernelContext* context, const OpInputList& values,
                        const OpInputList& input_mins,
                        const OpInputList& input_maxes,
                        const MklDnnShapeList& mkl_input_shapes,
                        bool quantized_input) {
    size_t num_mkl_input_shapes = mkl_input_shapes.size();
    DCHECK_EQ(values.size(), num_mkl_input_shapes);
    std::vector<Tensor> converted_values(num_mkl_input_shapes);
    TensorShapeList tf_input_shapes;
    for (size_t i = 0; i < num_mkl_input_shapes; ++i) {
      if (mkl_input_shapes[i].IsMklTensor()) {
        // Do conversion from MKL to TF
        OP_REQUIRES_OK(
            context, ConvertMklToTF<T>(context, values[i], mkl_input_shapes[i],
                                       &converted_values[i]));
        tf_input_shapes.push_back(mkl_input_shapes[i].GetTfShape());
      } else {
        // No conversion since it is TF tensor already
        converted_values[i] = values[i];
        tf_input_shapes.push_back(values[i].shape());
      }
    }

    // Call Eigen concat.
    eigen_concat_op_.Compute(context, converted_values, tf_input_shapes,
                             input_mins, input_maxes, quantized_input);

    if (!native_format) {
      // Get the number of dims from first input since all input tensors
      // should have same rank.
      size_t dims = values[0].shape().dims();
      MklDnnShape output_data_mkl_shape;
      output_data_mkl_shape.SetMklTensor(false);
      output_data_mkl_shape.SetDimensions(dims);
      AllocateOutputSetMklShape(context, 0, output_data_mkl_shape);
      if (quantized_input) {
        MklDnnShape output_min_max_mkl_shape;
        output_min_max_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(context, 1, output_min_max_mkl_shape);
        AllocateOutputSetMklShape(context, 2, output_min_max_mkl_shape);
      }
    }
  }

  // This method finds the most common format across all MKL inputs
  // Inputs:
  //   1. input_shapes: shapes of input (MKL) tensors.
  //   2. concat_dim: concat dimension.
  // Outputs:
  //   1. is_reorder_needed is set to true if inputs have difference formats
  //      It is set to false otherwise.
  //   2. concat_dim_size is the size of concat_dim.
  // Return:
  //   return the common MKL format.
  memory::format_tag FindMklCommonFormat(const MklDnnShapeList& input_shapes,
                                         int concat_dim,
                                         bool* is_reorder_needed,
                                         int64* concat_dim_size) {
    *is_reorder_needed = false;
    *concat_dim_size = 0;
    std::unordered_map<int, int> occurrence_map;
    if (input_shapes.size() == 0) return memory::format_tag::any;

    // Compute ocurrences of each format of all inputs.
    for (int k = 0; k < input_shapes.size(); k++) {
      auto src_dims = TFShapeToMklDnnDims(input_shapes[k].GetTfShape());
      *concat_dim_size += src_dims[concat_dim];
      int fmt = static_cast<int>(
          MklTensorFormatToMklDnnDataFormat(input_shapes[k].GetTfDataFormat()));
      occurrence_map[fmt] += 1;
    }

    if (occurrence_map.size() == 1) {
      // this means that all inputs have a same format
      // return it with is_reorder_needed set false.
      return static_cast<memory::format_tag>(
          MklTensorFormatToMklDnnDataFormat(input_shapes[0].GetTfDataFormat()));
    }

    // Input tensors have different formats. Thus, reorder is needed.
    // We pick up the most common format to minimize the total
    // number of input reorder.
    memory::format_tag commonest_format = memory::format_tag::any;
    int max_occurrence = 0;
    *is_reorder_needed = true;
    for (auto item : occurrence_map) {
      if (item.second > max_occurrence) {
        commonest_format = static_cast<memory::format_tag>(item.first);
        max_occurrence = item.second;
      }
    }
    return commonest_format;
  }
};

/* Use optimized concat for float type only */
#define REGISTER_MKL_CPU(type)                                 \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklConcat")                                       \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .HostMemory("concat_dim")                            \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklConcatOp<CPUDevice, type, NAME_IS_CONCAT_DIM>);       \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklConcatV2")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .TypeConstraint<int32>("Tidx")                       \
          .HostMemory("axis")                                  \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklConcatOp<CPUDevice, type, NAME_IS_AXIS>);

TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .HostMemory("axis")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklConcatOp<CPUDevice, quint8, NAME_IS_AXIS, true>);

REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConcatV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .HostMemory("axis")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklConcatOp<CPUDevice, qint8, NAME_IS_AXIS, true>);

#define REGISTER_QUANTIZED_CONCATV2(type)                \
  REGISTER_KERNEL_BUILDER(Name("QuantizedConcatV2")      \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("axis"),       \
                          NoOp)

REGISTER_QUANTIZED_CONCATV2(quint8);
REGISTER_QUANTIZED_CONCATV2(qint8);

#undef REGISTER_CONCAT_MKL
#undef CONCAT_PRIM_DESC
#undef CONCAT_PRIM_DESC_USING_SRC
#undef GET_MEMORY_DESC
#undef SET_MKL_LAYOUT
}  // namespace tensorflow

#endif  // INTEL_MKL
