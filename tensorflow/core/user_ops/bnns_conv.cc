#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/cast_op_impl.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

#include <Accelerate/Accelerate.h>


using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

bool BNNSActivationFunctionFromString(const string& activation_function_str, BNNSActivationFunction* bnnActivationFunction) {
  if (activation_function_str == "Linear") {
    *bnnActivationFunction = BNNSActivationFunctionIdentity;
    return true;
  } else if (activation_function_str == "ReLU") {
    *bnnActivationFunction = BNNSActivationFunctionRectifiedLinear;
    return true;
  }
  return false;
}

bool BNNSDataTypeFromString(const string& data_type_str, BNNSDataType* bnnsDataType) {
  if (data_type_str == "float32") {
    *bnnsDataType = BNNSDataTypeFloat32;
    return true;
  } else if (data_type_str == "float16") {
    *bnnsDataType = BNNSDataTypeFloat16;
    return true;
  }
  return false;
}


// Inspired by: https://github.com/tensorflow/tensorflow/blob/8746f8ac9e9ef652611180e0bf64466af2707b20/tensorflow/core/ops/nn_ops.cc#L503-L553
REGISTER_OP("Conv2DBNNS")
.Input("input: T")
.Input("filter: T")
.Output("output: T")
.Attr("T: {float}")
.Attr("strides: list(int) >= 4")
.Attr(GetPaddingAttrString())
.Attr(GetConvnetDataFormatAttrString())
.Attr("activation_function: {'Linear', 'ReLU'} = 'Linear'")
.Attr("weights_data_type: {'float32', 'float16'} = 'float32'")
.Attr("input_data_type: {'float32', 'float16'} = 'float32'")
.SetShapeFn(shape_inference::Conv2DShape);

REGISTER_OP("Conv2DBNNSWithBias")
.Input("input: T")
.Input("filter: T")
.Input("bias: T")
.Output("output: T")
.Attr("T: {float}")
.Attr("strides: list(int) >= 4")
.Attr(GetPaddingAttrString())
.Attr(GetConvnetDataFormatAttrString())
.Attr("activation_function: {'Linear', 'ReLU'} = 'Linear'")
.Attr("weights_data_type: {'float32', 'float16'} = 'float32'")
.Attr("input_data_type: {'float32', 'float16'} = 'float32'")
.SetShapeFn(shape_inference::Conv2DShape);

// Inspired by: https://github.com/tensorflow/tensorflow/blob/8746f8ac9e9ef652611180e0bf64466af2707b20/tensorflow/core/kernels/conv_ops.cc#L243-L391
// TODO: Fix usage of float rather than T
// Would be cool to handle half and ints
template <typename Device, typename T, bool biasEnabled>
class Conv2DBNNSOp : public OpKernel {
public:
    explicit Conv2DBNNSOp(OpKernelConstruction* context) : OpKernel(context) {
        const DataType dt = DataTypeToEnum<T>::v();
        if(biasEnabled) {
            OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt}));
        } else {
            OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
        }

        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
        string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                    errors::InvalidArgument("Invalid data format"));
        OP_REQUIRES(context, data_format_ == FORMAT_NCHW,
                    errors::InvalidArgument("Conv2DBNNS implementation only  "
                                            "supports NCHW tensor format for now."));
        OP_REQUIRES(context, strides_.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                            "specify 4 dimensions"));
        const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
        const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
        OP_REQUIRES(
                    context, stride_n == 1 && stride_c == 1,
                    errors::InvalidArgument("Current implementation does not yet support "
                                            "strides in the batch and depth dimensions."));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

        string activation_function;
        OP_REQUIRES_OK(context, context->GetAttr("activation_function", &activation_function));
        OP_REQUIRES(context, BNNSActivationFunctionFromString(activation_function, &bnnActivationFunction_),
                    errors::InvalidArgument("Invalid activation function"));

        string data_type_str;
        OP_REQUIRES_OK(context, context->GetAttr("weights_data_type", &data_type_str));
        OP_REQUIRES(context, BNNSDataTypeFromString(data_type_str, &weightsDataType_),
                    errors::InvalidArgument("Invalid weight data type"));

        OP_REQUIRES_OK(context, context->GetAttr("input_data_type", &data_type_str));
        OP_REQUIRES(context, BNNSDataTypeFromString(data_type_str, &inputDataType_),
                    errors::InvalidArgument("Invalid input data type"));
    }

    void Compute(OpKernelContext* context) override {
        // Input tensor is of the following dimensions:
        // [ batch, in_rows, in_cols, in_depth ]
        const Tensor& input = context->input(0);

        // Input filter is of the following dimensions:
        // [ filter_rows, filter_cols, in_depth, out_depth]
        const Tensor& filter = context->input(1);

        // For 2D convolution, there should be 4 dimensions.
        OP_REQUIRES(context, input.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            input.shape().DebugString()));
        OP_REQUIRES(context, filter.dims() == 4,
                    errors::InvalidArgument("filter must be 4-dimensional: ",
                                            filter.shape().DebugString()));

        for (int i = 0; i < 3; i++) {
            OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                                 std::numeric_limits<int>::max()),
                        errors::InvalidArgument("filter too large"));
        }

        // The last dimension for input is in_depth. It must be the same as the
        // filter's in_depth.
        const int64 in_depth = GetTensorDim(input, data_format_, 'C');
        OP_REQUIRES(
                    context, in_depth == filter.dim_size(2),
                    errors::InvalidArgument("input and filter must have the same depth: ",
                                            in_depth, " vs ", filter.dim_size(2)));

        // The last dimension for filter is out_depth.
        const int out_depth = static_cast<int>(filter.dim_size(3));

        // The second dimension for input is rows/height.
        // The first dimension for filter is rows/height.
        const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
        OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                             std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input rows too large"));
        const int input_rows = static_cast<int>(input_rows_raw);
        const int filter_rows = static_cast<int>(filter.dim_size(0));

        // The third dimension for input is columns/width.
        // The second dimension for filter is columns/width.
        const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
        OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                             std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input cols too large"));
        const int input_cols = static_cast<int>(input_cols_raw);
        const int filter_cols = static_cast<int>(filter.dim_size(1));

        // The first dimension for input is batch.
        const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
        OP_REQUIRES(context,
                    FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                    errors::InvalidArgument("batch is too large"));
        const int batch = static_cast<int>(batch_raw);

        // For now we take the stride from the second and third dimensions only (we
        // do not support striding on the batch or depth dimension).
        const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
        const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

        int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                             padding_, &out_rows, &pad_rows));
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                             padding_, &out_cols, &pad_cols));

        // This should have this format, the actual shape is fixed to match format
        TensorShape out_shape =
            ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

        // Output tensor is of the following dimensions:
        // [ in_batch, out_depth, out_rows, out_cols ]
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));


#ifdef TENSORFLOW_BNNS_PRINT_DEBUG
        // This was/ should be VLOG(2), but I was getting some weird link error
        std::cout << "BNNSConvOp"
                  << ": batch = " << batch
                  << ", in_depth = " << in_depth
                  << ", input_cols = " << input_cols
                  << ", filter_cols = " << filter_cols
                  << ", input_rows = " << input_rows
                  << ", filter_rows = " << filter_rows
                  << ", stride_rows = " << stride_rows
                  << ", stride_cols = " << stride_cols
                  << ", out_depth = " << out_depth
                  << ", out_cols = " << out_cols
                  << ", out_rows = " << out_rows
                  << std::endl;
#endif

        // If there is nothing to compute, return.
        if (out_shape.num_elements() == 0) {
            return;
        }

        // Description of the input image stack
        BNNSImageStackDescriptor i_desc;
        bzero(&i_desc,sizeof(i_desc));
        i_desc.width = input_cols;
        i_desc.height = input_rows;
        i_desc.channels = in_depth;
        i_desc.row_stride = input_cols;
        i_desc.image_stride = input_cols * input_rows;
        i_desc.data_type = inputDataType_;

        // Description of the output image stack
        BNNSImageStackDescriptor o_desc;
        bzero(&o_desc, sizeof(o_desc));
        o_desc.width = out_cols;
        o_desc.height = out_rows;
        o_desc.channels = out_depth;
        o_desc.row_stride = out_cols;
        o_desc.image_stride = out_cols * out_rows;
        o_desc.data_type = BNNSDataTypeFloat32;

        // Description of the convolution layer
        BNNSConvolutionLayerParameters layer_params;
        bzero(&layer_params, sizeof(layer_params));
        layer_params.k_width = filter_cols;
        layer_params.k_height = filter_rows;
        layer_params.in_channels = i_desc.channels;
        layer_params.out_channels = o_desc.channels;
        layer_params.x_stride = stride_cols;
        layer_params.y_stride = stride_rows;

        layer_params.x_padding = pad_cols;
        layer_params.y_padding = pad_rows;

        // [ outputChannel ][ inputChannel ][ kernelY ][ kernelX ]
        Tensor transformed_filter;
        {
            OP_REQUIRES_OK(context, context->allocate_temp(
                                                           DataTypeToEnum<T>::value,
                                                           TensorShape({out_depth, in_depth,
                filter_rows,
                filter_cols}),
                                                           &transformed_filter));

            functor::TransformFilter<Device, T, int, 4>()(
                                                          context->eigen_device<Device>(), To32Bit(filter.tensor<T, 4>()),
                                                          To32Bit(transformed_filter.tensor<T, 4>()));
        }

        layer_params.weights.data_type = weightsDataType_;

        // TODO: Use allocate_temp
        const void* weights;
        Tensor transformed_filter16;
        if (layer_params.weights.data_type == BNNSDataTypeFloat16) {
            Tensor transformed_filter16Temp(DT_HALF, transformed_filter.shape());
            GetCpuCastFromFloat(DT_HALF)(context, transformed_filter, &transformed_filter16Temp);
            transformed_filter16 = transformed_filter16Temp;
            weights = transformed_filter16.flat<Eigen::half>().data();
        } else {
            weights = transformed_filter.flat<T>().data();
        }

        // Attach weight buffer to layer parameters
        layer_params.weights.data = weights;

        // Attach bias buffer to layer parameters
        if (biasEnabled) {
            const Tensor& bias = context->input(2);

            OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                        errors::InvalidArgument("Biases must be 1D: ",
                                                bias.shape().DebugString()));

            OP_REQUIRES(
                context,
                bias.shape().dim_size(0) == out_depth,
                errors::InvalidArgument(
                    "Must provide as many biases as the last dimension "
                    "of the input tensor: ",
                    bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

            const T* biasVector = bias.flat<T>().data();

            layer_params.bias.data = biasVector;
            layer_params.bias.data_type = BNNSDataTypeFloat32;
        }

        layer_params.activation.function = bnnActivationFunction_;

        // Common filter parameters
        BNNSFilterParameters filter_params;
        bzero(&filter_params, sizeof(filter_params));

        // Create a new convolution layer filter
        // TODO: Figure out if we can keep this around to avoid the allocation / deallocation.
        // This should be possible if we know the sizes statically
        // and / or assume they won't change after first use. This might require a lock though for thread safety.

        BNNSFilter filter_bnns;
        {
           filter_bnns = BNNSFilterCreateConvolutionLayer(&i_desc, &o_desc, &layer_params, &filter_params);
        }
        OP_REQUIRES(context, filter_bnns != nullptr,
            errors::Unknown("BNNSFilterCreateConvolutionLayer failed"));

        // TODO: Use allocate_temp
        const void* i_stack;
        Tensor input16;
        if (i_desc.data_type == BNNSDataTypeFloat16) {
            Tensor input16Temp(DT_HALF, input.shape());
            GetCpuCastFromFloat(DT_HALF)(context, input, &input16Temp);
            input16 = input16Temp;
            i_stack = input16.flat<Eigen::half>().data();
        } else {
            i_stack = input.flat<T>().data();
        }

        T* o_stack = output->flat<T>().data();

        {
            // Apply filter to input stack. Result is written in output stack.
            int status = BNNSFilterApplyBatch(
                filter_bnns,
                batch,
                i_stack,
                input_cols * input_rows * in_depth,
                o_stack,
                out_cols * out_rows * out_depth);

            // TODO: Ideally use one of the error macros here though because we want to cleanup, would have to use
            // the OP_REQUIRES_ASYNC macro. That should be possible along with a lambda.
            if (!TF_PREDICT_TRUE(status == 0)) {
                context->CtxFailure(errors::Unknown("BNNSFilterApply failed"));
            }
        }

        {
            // Release resources
            BNNSFilterDestroy(filter_bnns);
        }
    }

private:
    std::vector<int32> strides_;
    Padding padding_;
    TensorFormat data_format_;
    BNNSActivationFunction bnnActivationFunction_;
    BNNSDataType weightsDataType_;
    BNNSDataType inputDataType_;

    TF_DISALLOW_COPY_AND_ASSIGN(Conv2DBNNSOp);

};

#define REGISTER_CPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
    Name("Conv2DBNNS").Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
    Conv2DBNNSOp<CPUDevice, T, false>);                                   \
  REGISTER_KERNEL_BUILDER(                                                \
    Name("Conv2DBNNSWithBias").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    Conv2DBNNSOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_CPU);
