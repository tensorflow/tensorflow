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

#include "mkldnn.h"
#include "mkldnn.hpp"
#include "mkldnn_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::primitive_attr;
using mkldnn::prop_kind;
using mkldnn::reorder;
using mkldnn::stream;

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

typedef Eigen::ThreadPoolDevice CPUDevice;

struct MklReorderWithScaleFwdParams {
  memory::dims src_dims;
  memory::desc src_md;
  memory::desc dst_md;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  PostOpParam post_op_params;

  MklReorderWithScaleFwdParams(memory::dims src_dims, memory::desc src_md,
                               memory::desc dst_md)
      : src_dims(src_dims), src_md(src_md), dst_md(dst_md) {}
};

class MklReorderWithScalePrimitive : public MklPrimitive {
 public:
  explicit MklReorderWithScalePrimitive(
      const memory* from, const memory* to,
      const MklReorderWithScaleFwdParams& fwdParams) {
    // Create reorder primitive
    Setup(from, to, fwdParams);
  }

  ~MklReorderWithScalePrimitive() {}

  std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }

  // set data handles
  void SetMemory(const memory* from, const memory* to) {
    context_.src_mem->set_data_handle(from->get_data_handle());
    context_.dst_mem->set_data_handle(to->get_data_handle());
  }

 private:
  // Primitive reuse context for reorder
  struct ReorderContext {
    // MKL-DNN memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;

    // Memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> dst_md;

    // Memory primitive desc
    std::shared_ptr<mkldnn::memory::primitive_desc> src_mpd;
    std::shared_ptr<mkldnn::memory::primitive_desc> dst_mpd;

    // Reorder primitive descriptor and primitive
    std::shared_ptr<reorder::primitive_desc> reorder_pd;
    std::shared_ptr<primitive> reorder_prim;

    ReorderContext()
        : src_mem(nullptr),
          dst_mem(nullptr),
          src_md(nullptr),
          dst_md(nullptr),
          src_mpd(nullptr),
          dst_mpd(nullptr),
          reorder_pd(nullptr),
          reorder_prim(nullptr) {}
  } context_;

  engine cpu_engine_ = engine(engine::cpu, 0);

  // Reorder primitive setup
  void Setup(const memory* from, const memory* to,
             const MklReorderWithScaleFwdParams& fwdParams) {
    // Create memory descriptors for reorder data with specified format
    context_.src_md.reset(new memory::desc(fwdParams.src_md.data));
    context_.dst_md.reset(new memory::desc(fwdParams.dst_md.data));
    context_.src_mpd.reset(
        new memory::primitive_desc(*context_.src_md, cpu_engine_));
    context_.dst_mpd.reset(
        new memory::primitive_desc(*context_.dst_md, cpu_engine_));

    // Check if there is any fusion as post-ops
    auto const& post_op_params = fwdParams.post_op_params;
    mkldnn::primitive_attr post_ops_attr;

    DCHECK(post_op_params.name == "scale");
    DCHECK_EQ(post_op_params.param.size(), 1);
    std::vector<float> scales;
    scales.push_back(post_op_params.param[0]);
    post_ops_attr.set_output_scales(0, scales);

    // Create a reorder
    context_.reorder_pd =
        std::make_shared<reorder::primitive_desc>(reorder::primitive_desc(
            *context_.src_mpd, *context_.dst_mpd, post_ops_attr));

    // Create memory primitive based on dummy data
    context_.src_mem.reset(new memory(*context_.src_mpd, DummyData));
    context_.dst_mem.reset(new memory(*context_.dst_mpd, DummyData));

    // Create reorder primitive
    context_.reorder_prim = std::make_shared<reorder>(
        reorder(*context_.reorder_pd, *context_.src_mem, *context_.dst_mem));
  }
};

template <typename T>
class MklReorderWithScalePrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklReorderWithScalePrimitive* Get(
      const memory* from, const memory* to,
      const MklReorderWithScaleFwdParams& fwdParams) {
    // Try to find a suitable primitive from the cached pool
    auto reorderPrim = static_cast<MklReorderWithScalePrimitive*>(
        MklReorderWithScalePrimitiveFactory<T>::GetInstance().GetReorder(
            from, to, fwdParams));
    if (reorderPrim == nullptr) {
      reorderPrim = new MklReorderWithScalePrimitive(from, to, fwdParams);
      MklReorderWithScalePrimitiveFactory<T>::GetInstance().SetReorder(
          from, to, reorderPrim, fwdParams);
    }
    reorderPrim->SetMemory(from, to);
    return reorderPrim;
  }

  static MklReorderWithScalePrimitiveFactory& GetInstance() {
    static MklReorderWithScalePrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklReorderWithScalePrimitiveFactory() {}
  ~MklReorderWithScalePrimitiveFactory() {}

  static string CreateKey(const memory* from, const memory* to,
                          const MklReorderWithScaleFwdParams& fwdParams) {
    string dtypes = string("");
    string prefix = "reorder";
    FactoryKeyCreator key_creator;
    auto const& from_desc = from->get_primitive_desc().desc().data;
    auto const& to_desc = to->get_primitive_desc().desc().data;

    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(static_cast<int>(from_desc.format));
    key_creator.AddAsKey(static_cast<int>(from_desc.data_type));
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey(static_cast<int>(to_desc.format));
    key_creator.AddAsKey(static_cast<int>(to_desc.data_type));
    key_creator.AddAsKey(fwdParams.dtypes);

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
};

// Fuction to find (or create) a reorder from memory pointed by
// 'from' to memory pointed by 'to', it will create primitive or
// get primitive from pool if it is cached.
// Returns the primitive.
template <typename T>
inline primitive FindOrCreateReorder(
    const memory* from, const memory* to,
    const MklReorderWithScaleFwdParams& fwdParams) {
  DCHECK(from);
  DCHECK(to);
  MklReorderWithScalePrimitive* reorder_prim =
      MklReorderWithScalePrimitiveFactory<T>::Get(from, to, fwdParams);
  return *reorder_prim->GetPrimitive();
}

// Quantizes a tensor from float to T, with user-specified min_range and
// max_range.
template <typename Device, typename T>
class MklQuantizeV2Op : public OpKernel {
 public:
  explicit MklQuantizeV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx, (mode_string == "MIN_COMBINED" ||
                      mode_string == "MIN_FIRST" || mode_string == "SCALED"),
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
    OP_REQUIRES(ctx, (round_mode_string == "HALF_AWAY_FROM_ZERO" ||
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

    auto cpu_engine = engine(engine::cpu, 0);
    const Tensor& input = ctx->input(0);
    const unsigned int src_idx = 0;
    const Tensor& src_tensor = MklGetInput(ctx, src_idx);

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);

    Tensor* output_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 0, &output_tensor, src_tensor.shape(),
                              output_mkl_shape);
    TensorShape min_tf_shape = {};
    MklDnnShape min_mkl_shape;
    min_mkl_shape.SetMklTensor(false);
    Tensor* output_min_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 1, &output_min_tensor, min_tf_shape,
                              min_mkl_shape);
    TensorShape max_tf_shape = {};
    MklDnnShape max_mkl_shape;
    max_mkl_shape.SetMklTensor(false);
    Tensor* output_max_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 2, &output_max_tensor, max_tf_shape,
                              max_mkl_shape);

    // Estimate scale for quantization
    float scale_factor = 0;
    const int number_of_bits = sizeof(T) * 8;
    const int64 number_of_steps = static_cast<int64>(1) << number_of_bits;
    scale_factor = (number_of_steps - 1.0) / (max_range - min_range);

    float* src_data = const_cast<float*>(src_tensor.flat<float>().data());
    T* out_data = output_tensor->flat<T>().data();

    out_data[0] = (src_data[0] - min_range) * scale_factor;
    output_min_tensor->flat<float>()(0) = min_range;
    output_max_tensor->flat<float>()(0) = max_range;

    return;
  }

  void Compute(OpKernelContext* ctx) override {
    const unsigned int src_idx = 0;
    const Tensor& input = ctx->input(src_idx);
    const float input_min_range = ctx->input(1).flat<float>()(0);
    const float input_max_range = ctx->input(2).flat<float>()(0);
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
    auto cpu_engine = engine(engine::cpu, 0);
    const Tensor& src_tensor = MklGetInput(ctx, src_idx);
    MklDnnShape src_mkl_shape;
    GetMklShape(ctx, src_idx, &src_mkl_shape);
    auto src_tf_shape = src_mkl_shape.IsMklTensor() ? src_mkl_shape.GetTfShape()
                                                    : src_tensor.shape();
    auto src_dims = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetSizesAsMklDnnDims()
                        : TFShapeToMklDnnDims(src_tensor.shape());
    auto output_dims = src_dims;
    // Set the dst layout to be the best mkl layout based on dims and type.
    memory::format dst_layout_type;
    switch (src_tf_shape.dims()) {
      case 0:
        ComputeScalar(ctx, min_range, max_range);
        return;
      case 1:
        dst_layout_type = memory::format::x;
        break;
      case 2:
        dst_layout_type = memory::format::nc;
        break;
      case 3:
        dst_layout_type = memory::format::tnc;
        break;
      case 4:
        dst_layout_type = memory::format::nhwc;
        break;
      case 5:
        dst_layout_type = memory::format::ndhwc;
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
    auto src_md =
        src_mkl_shape.IsMklTensor()
            ? src_mkl_shape.GetMklLayout()
            : memory::desc(src_dims, MklDnnType<float>(), dst_layout_type);

    // If the mode is min_first, input data has to be subtracted from
    // min_range, before being scaled
    auto flat_input = input.flat<float>().data();
    Tensor minfirst_tmpinput;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_FLOAT, input.shape(), &minfirst_tmpinput));
    if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
      auto minfirst_input = minfirst_tmpinput.flat<float>().data();
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

      src.SetUsrMem(src_md, minfirst_input);
    } else {
      src.SetUsrMem(src_md, &src_tensor);
    }

    memory::desc dst_md =
        memory::desc(src_dims, MklDnnType<T>(), dst_layout_type);
    auto dst_pd = src.GetUsrMemPrimDesc();
    // Standard shape assignments for layout pass
    MklDnnShape output_mkl_shape;
    TensorShape output_tf_shape;
    if (src_mkl_shape.IsMklTensor()) {
      output_mkl_shape.SetMklTensor(true);
      output_mkl_shape.SetMklLayout(&dst_md);
      output_mkl_shape.SetElemType(MklDnnType<T>());
      output_mkl_shape.SetTfLayout(src_mkl_shape.GetDimension(),
                                   src_mkl_shape.GetSizesAsMklDnnDims(),
                                   src_mkl_shape.GetTfDataFormat());
      output_tf_shape.AddDim(dst_pd.get_size() / sizeof(T));
    } else {
      output_mkl_shape.SetMklTensor(false);
      output_tf_shape = MklDnnDimsToTFShape(output_dims);
    }

    Tensor* output_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 0, &output_tensor, output_tf_shape,
                              output_mkl_shape);
    TensorShape min_tf_shape = {};
    MklDnnShape min_mkl_shape;
    min_mkl_shape.SetMklTensor(false);
    Tensor* output_min_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 1, &output_min_tensor, min_tf_shape,
                              min_mkl_shape);
    TensorShape max_tf_shape = {};
    MklDnnShape max_mkl_shape;
    max_mkl_shape.SetMklTensor(false);
    Tensor* output_max_tensor = nullptr;
    AllocateOutputSetMklShape(ctx, 2, &output_max_tensor, max_tf_shape,
                              max_mkl_shape);

    dst.SetUsrMem(dst_md, output_tensor);

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
        target_range = static_cast<float>((uint64_t{1} << (num_bits - 1)) - 1);
      } else {
        max_range = max_abs;
        min_range = 0.0;
        // If it is unsigned and num_bits == 8, the range with 8 bits is [0,
        // 255].  If the input range is [0, x], then the scale is 255/x instead
        // of 254 as in the case above.
        target_range = static_cast<float>((uint64_t{1} << num_bits) - 1);
      }
      scale_factor = target_range / max_abs;

      output_min_tensor->flat<float>()(0) = min_range;
      output_max_tensor->flat<float>()(0) = max_range;

      // Primitive creation and stream submit
      std::vector<float> scales{scale_factor};
      mkldnn::primitive_attr attr;
      attr.set_output_scales(0, scales);
      auto reorder_desc = reorder::primitive_desc(
          src.GetUsrMemPrimDesc(), dst.GetUsrMemPrimDesc(), attr);
      reorder my_reorder = reorder(
          reorder_desc, primitive::at(*src.GetUsrMem()), *dst.GetUsrMem());
      std::vector<primitive> net{my_reorder};
      stream(stream::kind::eager).submit(net).wait();
    } else if (mode_ == QUANTIZE_MODE_MIN_FIRST) {
      // Estimate scale for qunatization
      const int number_of_bits = sizeof(T) * 8;
      const int64 number_of_steps = static_cast<int64>(1) << number_of_bits;
      scale_factor = (number_of_steps - 1.0) / (max_range - min_range);

      output_min_tensor->flat<float>()(0) = min_range;
      output_max_tensor->flat<float>()(0) = max_range;

      MklReorderWithScaleFwdParams fwdParams(src_dims, src_md, dst_md);
      fwdParams.dtypes.append(typeid(T).name());

      fwdParams.post_op_params.name = "scale";
      fwdParams.post_op_params.param.push_back(scale_factor);

      // Get primitive from pool or create one and submit
      std::vector<primitive> net;
      net.push_back(
          FindOrCreateReorder<T>(src.GetUsrMem(), dst.GetUsrMem(), fwdParams));
      stream(stream::kind::eager).submit(net).wait();
    }
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
                        MklQuantizeV2Op<CPUDevice, quint8>);
REGISTER_KERNEL_BUILDER(Name("_MklQuantizeV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklQuantizeV2Op<CPUDevice, qint8>);
}  // namespace tensorflow

#endif  // INTEL_MKL
