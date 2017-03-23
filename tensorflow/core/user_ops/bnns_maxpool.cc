#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

#include <Accelerate/Accelerate.h>


using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

// Inspired by: https://github.com/tensorflow/tensorflow/blob/8746f8ac9e9ef652611180e0bf64466af2707b20/tensorflow/core/ops/nn_ops.cc#L1237-L1260
REGISTER_OP("MaxPoolBNNS")
    .Attr("T: {float} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::MaxPoolShape);

// Inspired by: https://github.com/tensorflow/tensorflow/blob/2098b9abcf20d2c9694055bbfd6997bc00b73578/tensorflow/core/kernels/pooling_ops_common.h#L71-L255
// TODO: Fix usage of float rather than T
// Would be cool to handle half and ints
// An implementation of MaxPooling (forward).
template <typename Device, typename T>
class MaxPoolingBNNSOp : public OpKernel {
 public:
  explicit MaxPoolingBNNSOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    auto status = context->GetAttr("data_format", &data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NCHW,
          errors::InvalidArgument("Default MaxPoolingBNNSOp only supports NCHW."));
    } else {
      data_format_ = FORMAT_NCHW;
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,  ksize_,      stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    OP_REQUIRES(context, params.depth_window == 1,
                errors::Unimplemented("Depthwise max pooling not supported"));
    OP_REQUIRES(context, params.window_cols == params.window_rows,
                errors::InvalidArgument("ksize must be square"));
    OP_REQUIRES(context, params.col_stride == params.row_stride,
                errors::InvalidArgument("strides must be square"));

    Tensor* output = nullptr;
    auto out_shape = params.forward_output_shape();
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, out_shape, &output));

#ifdef TENSORFLOW_BNNS_PRINT_DEBUG
    // This was/ should be VLOG(2), but I was getting some weird link error
    std::cout << "MaxPoolingBNNSOp"
              << ": batch = " << params.tensor_in_batch
              << ", in_depth = " << params.depth
              << ", input_cols = " << params.tensor_in_cols
              << ", window_cols = " << params.window_cols
              << ", input_rows = " << params.tensor_in_rows
              << ", window_rows = " << params.window_rows
              << ", stride_rows = " << params.row_stride
              << ", stride_cols = " << params.col_stride
              << ", out_depth = " << params.depth
              << ", out_cols = " << params.out_width
              << ", out_rows = " << params.out_height
              << std::endl;
#endif

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
        return;
    }

    // Description of the input image stack
    BNNSImageStackDescriptor i_desc;
    bzero(&i_desc,sizeof(i_desc));
    i_desc.width = params.tensor_in_cols;
    i_desc.height = params.tensor_in_rows;
    i_desc.channels = params.depth;
    // TODO: are these always right?
    i_desc.row_stride = params.tensor_in_cols;
    i_desc.image_stride = params.tensor_in_cols * params.tensor_in_rows;
    i_desc.data_type = BNNSDataTypeFloat32;       // pixels values are 'float'

    // Description of the output image stack
    BNNSImageStackDescriptor o_desc;
    bzero(&o_desc, sizeof(o_desc));
    o_desc.width = params.out_width;
    o_desc.height = params.out_height;
    o_desc.channels = params.depth;
    // TODO: are these always right?
    o_desc.row_stride = params.out_width;
    o_desc.image_stride = params.out_width * params.out_height;
    o_desc.data_type = BNNSDataTypeFloat32;       // pixels values are 'float'

    // Description of the pooling layer
    BNNSPoolingLayerParameters layer_params;
    bzero(&layer_params, sizeof(layer_params));
    layer_params.k_width = params.window_cols;
    layer_params.k_height = params.window_rows;
    layer_params.in_channels = params.depth;
    layer_params.out_channels = params.depth;
    layer_params.x_stride = params.col_stride;
    layer_params.y_stride = params.row_stride;
    layer_params.pooling_function = BNNSPoolingFunctionMax;
    layer_params.x_padding = params.pad_cols;
    layer_params.y_padding = params.pad_rows;

    // Common filter parameters
    BNNSFilterParameters filter_params;
    bzero(&filter_params, sizeof(filter_params));

    // Create a new pooling layer filter
    // TODO: Figure out if we can keep this around to avoid the allocation / deallocation.
    // This should be possible if we know the sizes statically
    // and / or assume they won't change after first use. This might require a lock though for thread safety.
    BNNSFilter filter_bnns = BNNSFilterCreatePoolingLayer(&i_desc, &o_desc, &layer_params, &filter_params);
    OP_REQUIRES(context, filter_bnns != nullptr,
        errors::Unknown("BNNSFilterCreatePoolingLayer failed"));

    const T* i_stack = tensor_in.flat<T>().data();
    T* o_stack = output->flat<T>().data();

    // Apply filter to input stack. Result is written in output stack.
    int status = BNNSFilterApplyBatch(
        filter_bnns,
        params.tensor_in_batch,
        i_stack,
        params.tensor_in_cols * params.tensor_in_rows * params.depth,
        o_stack,
        params.out_width * params.out_height * params.depth
    );

    // TODO: Ideally use one of the error macros here though because we want to cleanup, would have to use
    // the OP_REQUIRES_ASYNC macro. That should be possible along with a lambda.
    if (!TF_PREDICT_TRUE(status == 0)) {
        context->CtxFailure(errors::Unknown("BNNSFilterApply failed"));
    }

    // Release resources
    BNNSFilterDestroy(filter_bnns);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};


#define REGISTER_CPU(T)                                                   \
    REGISTER_KERNEL_BUILDER(                                              \
        Name("MaxPoolBNNS").Device(DEVICE_CPU).TypeConstraint<T>("T"),    \
        MaxPoolingBNNSOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU);
