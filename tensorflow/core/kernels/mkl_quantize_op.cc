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
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::primitive_attr;
using mkldnn::prop_kind;
using mkldnn::reorder;
using mkldnn::stream;

namespace {
enum { QUANTIZE_MODE_SCALED };
enum {
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};
}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Quantizes a tensor from float to T, with user-specified min_range and
// max_range.
template <typename Device, typename T>
class MklQuantizeV2Op : public OpKernel {
 public:
  explicit MklQuantizeV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx, (mode_string == "SCALED"),
                errors::InvalidArgument("mode must be scaled"));
    mode_ = QUANTIZE_MODE_SCALED;
    string round_mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("round_mode", &round_mode_string));
    OP_REQUIRES(ctx, (round_mode_string == "HALF_TO_EVEN"),
                errors::InvalidArgument("Round mode must be half to even"));
    round_mode_ = ROUND_HALF_TO_EVEN;
  }

  ~MklQuantizeV2Op() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const float input_min_range = ctx->input(1).flat<float>()(0);
    const float input_max_range = ctx->input(2).flat<float>()(0);
    float min_range = std::min(0.0f, input_min_range);
    float max_range;
    OP_REQUIRES(ctx, (input_max_range > input_min_range),
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
                                                  fabsf(input_max_range))) /
                          100.0f;
    max_range = std::max(input_max_range, min_range + epsilon);
    // Clamping the max_range to zero since max_range can also be negative.
    max_range = std::max(0.0f, max_range);
    auto cpu_engine = engine(engine::cpu, 0);
    const unsigned int src_idx = 0;
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
    src.SetUsrMem(src_md, &src_tensor);

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
    output_min_tensor->flat<float>()(0) = min_range;
    output_max_tensor->flat<float>()(0) = max_range;
    const float scale_factor = target_range / max_abs;
    // Primitive creation and stream submit
    std::vector<float> scales{scale_factor};
    mkldnn::primitive_attr attr;
    attr.set_output_scales(0, scales);
    auto reorder_desc = reorder::primitive_desc(src.GetUsrMemPrimDesc(),
                                                dst.GetUsrMemPrimDesc(), attr);
    reorder my_reorder = reorder(reorder_desc, primitive::at(*src.GetUsrMem()),
                                 *dst.GetUsrMem());
    std::vector<primitive> net{my_reorder};
    stream(stream::kind::eager).submit(net).wait();
  }

 private:
  int mode_;
  int round_mode_;
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
