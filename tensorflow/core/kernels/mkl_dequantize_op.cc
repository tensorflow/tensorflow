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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"

#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/util/mkl_util.h"

#include "mkldnn.hpp"
using mkldnn::primitive_attr;
using mkldnn::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklDequantizeOp : public OpKernel {
 public:
  explicit MklDequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string mode_string;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_string));
    OP_REQUIRES(ctx, mode_string == "SCALED",
                errors::InvalidArgument(
                    "MklDequantizeOp only supports 'SCALED' mode, but got '" +
                    mode_string + "'"));
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      // Using CPU device
      auto cpu_engine = engine(engine::cpu, 0);

      // Get the inputs
      const Tensor& src_tensor = MklGetInput(ctx, kSrcIndex);
      const float min_range =
          MklGetInput(ctx, kMinIndex).template flat<float>()(0);
      const float max_range =
          MklGetInput(ctx, kMaxIndex).template flat<float>()(0);

      // Get MklShape
      MklDnnShape src_mkl_shape;
      GetMklShape(ctx, kSrcIndex, &src_mkl_shape);

      // src_dims is the dimension of src_tensor
      // output_dims are same as src_dims
      auto src_dims = src_mkl_shape.IsMklTensor()
                          ? src_mkl_shape.GetSizesAsMklDnnDims()
                          : TFShapeToMklDnnDims(src_tensor.shape());
      auto output_dims = src_dims;

      // Create reorder memory for src and dst
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<float> dst(&cpu_engine);

      // If input is in MKL layout, then simply grab input layout; otherwise,
      // construct input TF layout. For TF layout, although input shape
      // (src_dims) required is in MKL-DNN order, the layout is Tensorflow's
      // layout
      auto src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<T>(), memory::format::nhwc);

      src.SetUsrMem(src_md, &src_tensor);

      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;

      memory::primitive_desc src_pd =
          memory::primitive_desc(src_md, cpu_engine);
      memory::desc dst_md = src_mkl_shape.IsMklTensor()
                                ? src_md
                                : memory::desc(src_dims, MklDnnType<float>(),
                                               memory::format::nhwc);
      memory::primitive_desc dst_pd =
          memory::primitive_desc(dst_md, cpu_engine);

      // If input is MKL shape, output is also MKL shape.
      // If input is TF shape, output is also TF shape.
      if (src_mkl_shape.IsMklTensor()) {
        output_mkl_shape.SetMklTensor(true);
        output_mkl_shape.SetMklLayout(&dst_pd);
        output_mkl_shape.SetElemType(MklDnnType<float>());
        output_mkl_shape.SetTfLayout(src_mkl_shape.GetDimension(),
                                     src_mkl_shape.GetSizesAsMklDnnDims(),
                                     src_mkl_shape.GetTfDataFormat());
        output_tf_shape.AddDim((dst_pd.get_size() / sizeof(float)));
      } else {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = MklDnnDimsToTFShape(output_dims);
      }

      // Allocate MKL or TF output shape based on the above
      AllocateOutputSetMklShape(ctx, 0, &output_tensor, output_tf_shape,
                                output_mkl_shape);
      dst.SetUsrMem(dst_md, output_tensor);

      // The quantization logic here for mode SCALED is similar to the logic
      // in QuantizeAndDequantizeV2 and QuantizeAndDequantizeV3.
      static constexpr int num_bits = sizeof(T) * 8;
      const float max_abs = std::max(std::abs(min_range), std::abs(max_range));
      bool is_signed = std::is_signed<T>::value;
      // If it is signed, we try to keep 0.0 being 0 and drop one bucket. For
      // example, if it is 8 bits, we have the range [-127, 127]. So for input
      // range of [-x, x], the scale should be (2*x)/254.
      //
      // If it is unsigned and num_bits == 8, the range with 8 bits is [0, 255].
      // If the input range is [0, x], then the scale is x/255 instead of 254 as
      // in the case above.
      const int target_bits = is_signed ? (num_bits - 1) : num_bits;
      const float target_range =
          static_cast<float>((uint64_t{1} << target_bits) - 1);
      const float scale_factor = max_abs / target_range;

      std::vector<float> scales;
      scales.push_back(scale_factor);
      primitive_attr attr;
      attr.set_output_scales(0, scales);
      attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
      mkldnn::reorder::primitive_desc reorder_pd =
          mkldnn::reorder::primitive_desc(src_pd, dst_pd, attr);

      // Execute MKL-DNN primitive
      std::vector<primitive> net;
      net.push_back(
          mkldnn::reorder(reorder_pd, *src.GetUsrMem(), *dst.GetUsrMem()));
      stream(stream::kind::eager).submit(net).wait();
    } catch (mkldnn::error& e) {
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
};

REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, quint8>);
REGISTER_KERNEL_BUILDER(Name("_MklDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklDequantizeOp<CPUDevice, qint8>);

}  // namespace tensorflow

#endif  // INTEL_MKL
