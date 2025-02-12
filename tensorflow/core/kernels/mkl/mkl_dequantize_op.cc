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

#if defined(INTEL_MKL)

#define EIGEN_USE_THREADS

#include <limits>

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::primitive_attr;
using dnnl::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, typename U, bool native_format = false>
class MklDequantizeOp : public OpKernel {
 public:
  explicit MklDequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx, mode_string == "SCALED",
                errors::InvalidArgument(
                    "MklDequantizeOp only supports 'SCALED' mode, but got '" +
                    mode_string + "'"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES(
        ctx,
        (ctx->output_type(0) == DT_FLOAT || ctx->output_type(0) == DT_BFLOAT16),
        errors::InvalidArgument("Output type must be float or bfloat16,"
                                " is '" +
                                DataTypeString(ctx->output_type(0)) + "'"));
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      // Using CPU device
      auto cpu_engine = engine(engine::kind::cpu, 0);

      // Get the inputs
      const Tensor& src_tensor = ctx->input(kSrcIndex);
      const float min_range = ctx->input(kMinIndex).template scalar<float>()();
      const float max_range = ctx->input(kMaxIndex).template scalar<float>()();

      // Get MklShape
      auto src_tf_shape = src_tensor.shape();

      // src_dims is the dimension of src_tensor
      // output_dims are same as src_dims
      auto src_dims = TFShapeToMklDnnDims(src_tensor.shape());
      auto output_dims = src_dims;

      // Create reorder memory for src and dst
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<U> dst(&cpu_engine);

      std::shared_ptr<stream> reorder_stream;
      // Create the oneDNN wrapper over Eigen threadpool and set max threads
      // in oneDNN.
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(ctx);
      tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                     ThreadPoolUseCallerThread());
      reorder_stream.reset(CreateStream(&eigen_tp, cpu_engine));

      memory::format_tag dst_layout_type;
      switch (src_tf_shape.dims()) {
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
          OP_REQUIRES_OK(
              ctx, errors::InvalidArgument("Input dims must be <= 5 and >= 1"));
          return;
      }

      // If input is in MKL layout, then simply grab input layout; otherwise,
      // construct input TF layout. For TF layout, although input shape
      // (src_dims) required is in MKL-DNN order, the layout is Tensorflow's
      // layout
      auto src_md = memory::desc(src_dims, MklDnnType<T>(), dst_layout_type);

      src.SetUsrMem(src_md, &src_tensor);
      src.SetUsrMemDataHandle(&src_tensor, reorder_stream);

      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;
      memory::desc dst_md =
          memory::desc(src_dims, MklDnnType<U>(), dst_layout_type);

      // If input is TF shape, output is also TF shape.
      output_mkl_shape.SetMklTensor(false);
      output_tf_shape = MklDnnDimsToTFShape(output_dims);

      // Allocate MKL or TF output shape based on the above
      AllocateOutputSetMklShape(ctx, 0, &output_tensor, output_tf_shape,
                                output_mkl_shape, native_format);
      dst.SetUsrMem(dst_md, output_tensor);
      dst.SetUsrMemDataHandle(output_tensor, reorder_stream);

      // The quantization logic here for mode SCALED is similar to the logic
      // in QuantizeAndDequantizeV2 and QuantizeAndDequantizeV3.
      static constexpr int num_bits = sizeof(T) * 8;
      bool is_signed = std::numeric_limits<T>::is_signed;

      const int target_bits = is_signed ? (num_bits - 1) : num_bits;
      const float v_max = static_cast<float>(uint64_t{1} << target_bits) - 1;
      float v_min = 0;
      if (is_signed) {
        v_min = -(static_cast<float>(uint64_t{1} << target_bits));
      }
      if (narrow_range_) {
        v_min += 1;
      }
      float scale_factor;
      if (v_min != 0) {
        scale_factor = std::max(min_range / v_min, max_range / v_max);
      } else {
        scale_factor = max_range / v_max;
      }
      std::vector<float> scales = {scale_factor};
      primitive_attr attr;
#ifndef ENABLE_ONEDNN_V3
      attr.set_output_scales(0, scales);
#else
      attr.set_scales_mask(DNNL_ARG_SRC, 0);
      auto scale_mem = memory({{static_cast<int64_t>(scales.size())},
                               MklDnnType<float>(),
                               memory::format_tag::x},
                              cpu_engine, scales.data());
#endif  // !ENABLE_ONEDNN_V3

      // Create reorder primitive and then execute.
      auto reorder_pd =
          ReorderPd(cpu_engine, src.GetUsrMem()->get_desc(), cpu_engine,
                    dst.GetUsrMem()->get_desc(), attr);
      std::vector<primitive> net = {reorder(reorder_pd)};
      std::vector<std::unordered_map<int, memory>> reorder_net_args;
#ifndef ENABLE_ONEDNN_V3
      reorder_net_args.push_back({{DNNL_ARG_FROM, *src.GetUsrMem()},
                                  { DNNL_ARG_TO,
                                    *dst.GetUsrMem() }});
#else
      reorder_net_args.push_back(
          {{DNNL_ARG_FROM, *src.GetUsrMem()},
           {DNNL_ARG_TO, *dst.GetUsrMem()},
           {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_mem}});
#endif  // !ENABLE_ONEDNN_V3
      execute_primitives(net, reorder_stream, reorder_net_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  const size_t kSrcIndex = 0;
  const size_t kMinIndex = 1;
  const size_t kMaxIndex = 2;
  bool narrow_range_;
};

REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .TypeConstraint<float>("dtype")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, quint8, float, true>);
REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .TypeConstraint<float>("dtype")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, qint8, float, true>);
REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .TypeConstraint<bfloat16>("dtype")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, quint8, bfloat16, true>);
REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .TypeConstraint<bfloat16>("dtype")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, qint8, bfloat16, true>);

}  // namespace tensorflow

#endif  // INTEL_MKL
