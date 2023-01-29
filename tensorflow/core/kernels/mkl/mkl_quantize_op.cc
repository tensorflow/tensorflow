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

#ifdef INTEL_MKL

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::primitive_attr;
using dnnl::prop_kind;
using dnnl::reorder;
using dnnl::stream;

namespace {
enum {
  QUANTIZE_MODE_MIN_COMBINED,
  QUANTIZE_MODE_MIN_FIRST,
  QUANTIZE_MODE_SCALED,
};
enum {
  // Round half away from zero: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5 if y > 0
  // round(y) = y - 0.5 if y < 0
  // E.g., -5.5 gets rounded to -6, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_AWAY_FROM_ZERO,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};
}  // namespace

namespace tensorflow {

#ifndef ENABLE_ONEDNN_V3
#define SET_MKL_LAYOUT(md) SetMklLayout(&md)
#else
#define SET_MKL_LAYOUT(md) SetMklLayout(md)
#endif  // !ENABLE_ONEDNN_V3

typedef Eigen::ThreadPoolDevice CPUDevice;

struct MklReorderWithScaleFwdParams {
  memory::dims src_dims;
  memory::desc src_md;
  memory::desc dst_md;
#ifdef ENABLE_ONEDNN_V3
  memory::desc scale_md;
#endif  // ENABLE_ONEDNN_V3
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  PostOpParam post_op_params;

#ifndef ENABLE_ONEDNN_V3
  MklReorderWithScaleFwdParams(memory::dims src_dims, memory::desc src_md,
                               memory::desc dst_md)
      : src_dims(src_dims), src_md(src_md), dst_md(dst_md) {}
#else
  MklReorderWithScaleFwdParams(memory::dims src_dims, memory::desc src_md,
                               memory::desc dst_md, memory::desc scale_md)
      : src_dims(src_dims),
        src_md(src_md),
        dst_md(dst_md),
        scale_md(scale_md) {}
#endif  // ENABLE_ONEDNN_V3
};

class MklReorderWithScalePrimitive : public MklPrimitive {
 public:
  explicit MklReorderWithScalePrimitive(
      const MklReorderWithScaleFwdParams& fwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create reorder primitive
    Setup(fwdParams);
  }

  ~MklReorderWithScalePrimitive() {}

  std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }

  void Execute(void* src_data, void* dst_data,
#ifdef ENABLE_ONEDNN_V3
               void* scale_data,
#endif  // ENABLE_ONEDNN_V3
               std::shared_ptr<stream> reorder_stream) {
#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
    context_.src_mem->set_data_handle(src_data, *reorder_stream);
    context_.dst_mem->set_data_handle(dst_data, *reorder_stream);
#else
    context_.src_mem->set_data_handle(src_data);
    context_.dst_mem->set_data_handle(dst_data);
#ifdef ENABLE_ONEDNN_V3
    context_.scale_mem->set_data_handle(scale_data);
#endif  // ENABLE_ONEDNN_V3
#endif  // !ENABLE_ONEDNN_OPENMP && !ENABLE_ONEDNN_V3
    context_.reorder_prim->execute(*reorder_stream, context_.prim_args);
    // After execution, set data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
#ifdef ENABLE_ONEDNN_V3
    context_.scale_mem->set_data_handle(DummyData);
#endif  // !ENABLE_ONEDNN_V3
  }

 private:
  // Primitive reuse context for reorder
  struct ReorderContext {
    // MKL-DNN memory
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> dst_mem;
#ifdef ENABLE_ONEDNN_V3
    std::shared_ptr<dnnl::memory> scale_mem;
#endif  // ENABLE_ONEDNN_V3

    // Reorder primitive descriptor and primitive
    std::shared_ptr<reorder::primitive_desc> reorder_pd;
    std::shared_ptr<primitive> reorder_prim;

    // Stream and primitive vector
    std::shared_ptr<dnnl::stream> reorder_stream;

    std::unordered_map<int, dnnl::memory> prim_args;

    ReorderContext()
        : src_mem(nullptr),
          dst_mem(nullptr),
#ifdef ENABLE_ONEDNN_V3
          scale_mem(nullptr),
#endif  // ENABLE_ONEDNN_V3
          reorder_pd(nullptr),
          reorder_prim(nullptr) {}
  } context_;

  // Reorder primitive setup
  void Setup(const MklReorderWithScaleFwdParams& fwdParams) {
    // Create memory descriptors for reorder data with specified format
    context_.src_mem.reset(
        new memory(fwdParams.src_md, cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(fwdParams.dst_md, cpu_engine_, DummyData));
#ifdef ENABLE_ONEDNN_V3
    context_.scale_mem.reset(
        new memory(fwdParams.scale_md, cpu_engine_, DummyData));
#endif  // ENABLE_ONEDNN_V3

    // Check if there is any fusion as post-ops
    dnnl::primitive_attr post_ops_attr;
#ifndef ENABLE_ONEDNN_V3
    auto const& post_op_params = fwdParams.post_op_params;
    DCHECK(post_op_params.name == "scale");
    DCHECK_EQ(post_op_params.param.size(), 1);
    std::vector<float> scales;
    scales.push_back(post_op_params.param[0]);
    post_ops_attr.set_output_scales(0, scales);
#else
    post_ops_attr.set_scales_mask(DNNL_ARG_DST, 0 /* mask */);
#endif  // !ENABLE_ONEDNN_V3

    context_.reorder_pd.reset(
        new ReorderPd(cpu_engine_, context_.src_mem->get_desc(), cpu_engine_,
                      context_.dst_mem->get_desc(), post_ops_attr));

    // Create reorder primitive
    context_.reorder_prim.reset(new reorder(*context_.reorder_pd));
    context_.prim_args.insert({DNNL_ARG_FROM, *context_.src_mem});
    context_.prim_args.insert({DNNL_ARG_TO, *context_.dst_mem});
#ifdef ENABLE_ONEDNN_V3
    context_.prim_args.insert(
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, *context_.scale_mem});
#endif  // ENABLE_ONEDNN_V3
  }

#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T>
class MklReorderWithScalePrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklReorderWithScalePrimitive* Get(
      const memory* from, const memory* to,
      const MklReorderWithScaleFwdParams& fwdParams) {
#ifndef ENABLE_ONEDNN_V3
    // Try to find a suitable primitive from the cached pool
    auto reorderPrim = static_cast<MklReorderWithScalePrimitive*>(
        MklReorderWithScalePrimitiveFactory<T>::GetInstance().GetReorder(
            from, to, fwdParams));
    if (reorderPrim == nullptr) {
      reorderPrim = new MklReorderWithScalePrimitive(fwdParams);
      MklReorderWithScalePrimitiveFactory<T>::GetInstance().SetReorder(
          from, to, reorderPrim, fwdParams);
    }
    return reorderPrim;
#else
    // TODO(intel-tf): enable ReorderWithScale primitive cache for v3.x
    auto reorderPrim = new MklReorderWithScalePrimitive(fwdParams);
    return reorderPrim;
#endif  // !ENABLE_ONEDNN_V3
  }

#ifndef ENABLE_ONEDNN_V3
  static MklReorderWithScalePrimitiveFactory& GetInstance() {
    static MklReorderWithScalePrimitiveFactory instance_;
    return instance_;
  }
#endif  // !ENABLE_ONEDNN_V3

 private:
  MklReorderWithScalePrimitiveFactory() {}
  ~MklReorderWithScalePrimitiveFactory() {}

#ifndef ENABLE_ONEDNN_V3
  static string CreateKey(const memory* from, const memory* to,
                          const MklReorderWithScaleFwdParams& fwdParams) {
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(MklReorderPrimitiveFactory<T>::CreateKey(from, to));
    // Generate key for post-op scale
    if (fwdParams.post_op_params.name == "scale") {
      DCHECK_EQ(fwdParams.post_op_params.param.size(), 1);
      key_creator.AddAsKey(fwdParams.post_op_params.name);
      key_creator.AddAsKey(fwdParams.post_op_params.param[0]);
    } else {
      return string("not_a_key");
    }

    return key_creator.GetKey();
  }

  MklPrimitive* GetReorder(const memory* from, const memory* to,
                           const MklReorderWithScaleFwdParams& fwdParams) {
    string key = CreateKey(from, to, fwdParams);
    return this->GetOp(key);
  }

  void SetReorder(const memory* from, const memory* to, MklPrimitive* op,
                  const MklReorderWithScaleFwdParams& fwdParams) {
    string key = CreateKey(from, to, fwdParams);
    this->SetOp(key, op);
  }
#endif  // !ENABLE_ONEDNN_V3
};

// Quantizes a tensor from float to T, with user-specified min_range and
// max_range.
template <typename Device, typename T, bool native_format = false>
class MklQuantizeV2Op : public OpKernel {
 public:
  explicit MklQuantizeV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx,
                (mode_string == "MIN_COMBINED" || mode_string == "MIN_FIRST" ||
                 mode_string == "SCALED"),
                errors::InvalidArgument("Mode string must be 'MIN_COMBINED',"
                                        " 'MIN_FIRST', or 'SCALED', is '" +
                                        mode_string + "'"));
    if (mode_string == "MIN_COMBINED") {
      mode_ = QUANTIZE_MODE_MIN_COMBINED;
    } else if (mode_string == "MIN_FIRST") {
      mode_ = QUANTIZE_MODE_MIN_FIRST;
    } else if (mode_string == "SCALED") {
      mode_ = QUANTIZE_MODE_SCALED;
    }

    string round_mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("round_mode", &round_mode_string));
    OP_REQUIRES(ctx,
                (round_mode_string == "HALF_AWAY_FROM_ZERO" ||
                 round_mode_string == "HALF_TO_EVEN"),
                errors::InvalidArgument("Round mode string must be "
                                        "'HALF_AWAY_FROM_ZERO' or "
                                        "'HALF_TO_EVEN', is '" +
                                        round_mode_string + "'"));
    if (round_mode_string == "HALF_AWAY_FROM_ZERO") {
      round_mode_ = ROUND_HALF_AWAY_FROM_ZERO;
    } else if (round_mode_string == "HALF_TO_EVEN") {
      OP_REQUIRES(ctx, mode_string == "SCALED",
                  errors::InvalidArgument("Round mode 'HALF_TO_EVEN' "
                                          "only supported for mode 'SCALED', "
                                          "but mode is '" +
                                          mode_string + "'."));
      round_mode_ = ROUND_HALF_TO_EVEN;
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("ensure_minimum_range", &ensure_minimum_range_));
  }

  void ComputeScalar(OpKernelContext* ctx, float min_range, float max_range) {
    // TODO(intel-tf): Scalar support has to be added for SCALE mode
    OP_REQUIRES(ctx, (mode_ == QUANTIZE_MODE_MIN_FIRST),
                errors::InvalidArgument(
                    "Scalar calculation in MKL is supported only for"
                    "MIN_FIRST mode for now."));

    // Min and max values of input range should be scalar.
    const Tensor& min_tensor = ctx->input(1);
    const Tensor& max_tensor = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(min_tensor.shape()),
        errors::InvalidArgument("`min_input` must be rank 0 but is rank ",
                                min_tensor.dims()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(max_tensor.shape()),
        errors::InvalidArgument("`max_input` must be rank 0 but is rank ",
                                max_tensor.dims()));

    auto cpu_engine = engine(engine::kind::cpu, 0);
    const unsigned int src_idx = 0;
    const Tensor& src_tensor = MklGetInput(ctx, src_idx);

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);

    Tensor* output_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 0, &output_tensor, src_tensor.shape(),
                              output_mkl_shape, native_format);
    TensorShape min_tf_shape = {};
    MklDnnShape min_mkl_shape;
    min_mkl_shape.SetMklTensor(false);
    Tensor* output_min_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 1, &output_min_tensor, min_tf_shape,
                              min_mkl_shape, native_format);
    TensorShape max_tf_shape = {};
    MklDnnShape max_mkl_shape;
    max_mkl_shape.SetMklTensor(false);
    Tensor* output_max_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 2, &output_max_tensor, max_tf_shape,
                              max_mkl_shape, native_format);

    // Estimate scale for quantization
    float scale_factor = 0;
    const int number_of_bits = sizeof(T) * 8;
    const int64 number_of_steps = static_cast<int64_t>(1) << number_of_bits;
    scale_factor = (number_of_steps - 1.0) / (max_range - min_range);

    float* src_data = const_cast<float*>(src_tensor.flat<float>().data());
    T* out_data = output_tensor->flat<T>().data();

    out_data[0] = (src_data[0] - min_range) * scale_factor;
    output_min_tensor->scalar<float>()() = min_range;
    output_max_tensor->scalar<float>()() = max_range;

    return;
  }

  void Compute(OpKernelContext* ctx) override {
    const unsigned int src_idx = 0;
    const Tensor& input = ctx->input(src_idx);
    const float input_min_range = ctx->input(1).scalar<float>()();
    const float input_max_range = ctx->input(2).scalar<float>()();
    float min_range = std::min(0.0f, input_min_range);
    float max_range;
    OP_REQUIRES(ctx, (input_max_range >= input_min_range),
                errors::InvalidArgument(
                    "input_max_range must be larger than input_min_range."));

    // When the minimum and maximum ranges are too close together, nudge them
    // apart by a small value so that they are slightly different. This helps
    // us avoid creating ill-formed buffers where all quantized values map to
    // the same float number. These kinds of buffers cause problems for
    // downstream ops when they need to do calculations on them.
    // We pick the value by making sure that zero is not more than 100x the
    // overall range from the maximum, so that the value can be easily
    // represented when we promote the quantized value to a higher
    // intermediate bit depth, since that's a common requirement.
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                  fabsf(input_max_range))) *
                          ensure_minimum_range_;
    max_range = std::max(input_max_range, min_range + epsilon);
    // Clamping the max_range to zero since max_range can also be negative.
    max_range = std::max(0.0f, max_range);
    auto cpu_engine = engine(engine::kind::cpu, 0);
    const Tensor& src_tensor = MklGetInput(ctx, src_idx);
    MklDnnShape src_mkl_shape;
    GetMklShape(ctx, src_idx, &src_mkl_shape, native_format);
    auto src_tf_shape = src_mkl_shape.IsMklTensor() ? src_mkl_shape.GetTfShape()
                                                    : src_tensor.shape();
    auto src_dims = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetSizesAsMklDnnDims()
                        : TFShapeToMklDnnDims(src_tensor.shape());
    auto output_dims = src_dims;
    // Set the dst layout to be the best mkl layout based on dims and type.
    memory::format_tag dst_layout_type;
    switch (src_tf_shape.dims()) {
      case 0:
        ComputeScalar(ctx, min_range, max_range);
        return;
      case 1:
        dst_layout_type = memory::format_tag::x;
        break;
      case 2:
        dst_layout_type = memory::format_tag::nc;
        break;
      case 3:
        dst_layout_type = memory::format_tag::tnc;
        break;
      case 4:
        dst_layout_type = memory::format_tag::nhwc;
        break;
      case 5:
        dst_layout_type = memory::format_tag::ndhwc;
        break;
      default:
        OP_REQUIRES_OK(ctx,
                       errors::Aborted("Input dims must be <= 5 and >= 1"));
        return;
    }
    // Create reorder memory for src, dst: both are defined in mkl_util.h,
    // they are wrapper
    MklDnnData<float> src(&cpu_engine);
    MklDnnData<T> dst(&cpu_engine);
#ifdef ENABLE_ONEDNN_V3
    MklDnnData<float> scale(&cpu_engine);
#endif  // ENABLE_ONEDNN_V3

    auto src_md =
        src_mkl_shape.IsMklTensor()
            ? src_mkl_shape.GetMklLayout()
            : memory::desc(src_dims, MklDnnType<float>(), dst_layout_type);

    // If the mode is min_first, input data has to be subtracted from
    // min_range, before being scaled
    auto flat_input = input.flat<float>().data();
    Tensor min_shifted_input_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, input.shape(),
                                           &min_shifted_input_tensor));
    if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
      auto minfirst_input = min_shifted_input_tensor.flat<float>().data();
      const Eigen::TensorOpCost cost(
          sizeof(float), /*load bytes*/
          sizeof(float), /*saved bytes*/
          Eigen::TensorOpCost::AddCost<float>() /*sub cost*/);

      const CPUDevice& d = ctx->eigen_device<CPUDevice>();
      auto ParallelSub = [&](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          minfirst_input[i] = flat_input[i] - min_range;
        }
      };
      d.parallelFor(input.NumElements(), cost, ParallelSub);

      src.SetUsrMem(src_md, &min_shifted_input_tensor);
    } else {
      src.SetUsrMem(src_md, &src_tensor);
    }

    memory::desc dst_md =
        memory::desc(src_dims, MklDnnType<T>(), dst_layout_type);

    // Standard shape assignments for layout pass
    MklDnnShape output_mkl_shape;
    TensorShape output_tf_shape;
    if (src_mkl_shape.IsMklTensor()) {
      output_mkl_shape.SetMklTensor(true);
      output_mkl_shape.SET_MKL_LAYOUT(dst_md);
      output_mkl_shape.SetElemType(MklDnnType<T>());
      output_mkl_shape.SetTfLayout(src_mkl_shape.GetDimension(),
                                   src_mkl_shape.GetSizesAsMklDnnDims(),
                                   src_mkl_shape.GetTfDataFormat());
      output_tf_shape.AddDim(dst_md.get_size() / sizeof(T));
    } else {
      output_mkl_shape.SetMklTensor(false);
      output_tf_shape = MklDnnDimsToTFShape(output_dims);
    }

    Tensor* output_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 0, &output_tensor, output_tf_shape,
                              output_mkl_shape, native_format);
    dst.SetUsrMem(dst_md, output_tensor);

    TensorShape min_tf_shape = {};
    MklDnnShape min_mkl_shape;
    min_mkl_shape.SetMklTensor(false);
    Tensor* output_min_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 1, &output_min_tensor, min_tf_shape,
                              min_mkl_shape, native_format);
    TensorShape max_tf_shape = {};
    MklDnnShape max_mkl_shape;
    max_mkl_shape.SetMklTensor(false);
    Tensor* output_max_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 2, &output_max_tensor, max_tf_shape,
                              max_mkl_shape, native_format);

    float scale_factor = 0;
    if (mode_ == QUANTIZE_MODE_SCALED) {
      // Estimating scales for quantization.
      const int num_bits = sizeof(T) * 8;
      const float max_abs = std::max(std::abs(min_range), std::abs(max_range));
      const bool is_signed = std::is_signed<T>::value;
      float target_range;
      if (is_signed) {
        max_range = max_abs;
        min_range = -max_abs;
        // If it is signed, we try to keep 0.0 being 0 and drop one bucket. For
        // example, if it is 8 bits, we have the range [-127, 127]. So for input
        // range of [-x, x], the scale should be 254/(2*x).
        target_range = static_cast<float>((uint64_t{1} << num_bits) - 1) / 2.;
      } else {
        max_range = max_abs;
        min_range = 0.0;
        // If it is unsigned and num_bits == 8, the range with 8 bits is [0,
        // 255].  If the input range is [0, x], then the scale is 255/x instead
        // of 254 as in the case above.
        target_range = static_cast<float>((uint64_t{1} << num_bits) - 1);
      }
      scale_factor = target_range / max_abs;
    } else if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
      // Estimate scale for qunatization
      const int number_of_bits = sizeof(T) * 8;
      const int64 number_of_steps = static_cast<int64_t>(1) << number_of_bits;
      scale_factor = (number_of_steps - 1.0) / (max_range - min_range);
    }
#ifdef ENABLE_ONEDNN_V3
    auto scale_md =
        memory::desc({1}, MklDnnType<float>(), memory::format_tag::x);
    MklReorderWithScaleFwdParams fwdParams(src_dims, src_md, dst_md, scale_md);
    Tensor scale_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, {1}, &scale_tensor));
    scale_tensor.flat<float>()(0) = 1 / scale_factor;
    scale.SetUsrMem(scale_md, &scale_tensor);
#else
    MklReorderWithScaleFwdParams fwdParams(src_dims, src_md, dst_md);
    fwdParams.dtypes.append(typeid(T).name());
    fwdParams.post_op_params.name = "scale";
    fwdParams.post_op_params.param.push_back(scale_factor);
#endif  // ENABLE_ONEDNN_V3

    MklDnnThreadPool eigen_tp(ctx);
    MklReorderWithScalePrimitive* reorder_prim =
        MklReorderWithScalePrimitiveFactory<T>::Get(src.GetUsrMem(),
                                                    dst.GetUsrMem(), fwdParams);
    std::shared_ptr<stream> cpu_stream;

    cpu_stream.reset(CreateStream(&eigen_tp, reorder_prim->GetEngine()));
    reorder_prim->Execute(src.GetUsrMemDataHandle(), dst.GetUsrMemDataHandle(),
#ifdef ENABLE_ONEDNN_V3
                          scale.GetUsrMemDataHandle(),
#endif  // ENABLE_ONEDNN_V3
                          cpu_stream);

    output_min_tensor->scalar<float>()() = min_range;
    output_max_tensor->scalar<float>()() = max_range;
  }

 private:
  float ensure_minimum_range_;
  int mode_;
  int round_mode_;
  int axis_;
  bool narrow_range_;
};

REGISTER_KERNEL_BUILDER(Name("_MklQuantizeV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklQuantizeV2Op<CPUDevice, quint8, true>);
REGISTER_KERNEL_BUILDER(Name("_MklQuantizeV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklQuantizeV2Op<CPUDevice, qint8, true>);

#undef SET_MKL_LAYOUT

}  // namespace tensorflow

#endif  // INTEL_MKL
