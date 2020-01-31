#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/kernels/xcore/xcore_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {


XcoreOpsResolver::XcoreOpsResolver() {
    // By default we register all tflite builtin operators
    // To decrease the size of the runtime, comment out unused builtin operators
    AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(), 1, 4);
    AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D());
    AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX());
    AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC());
    AddBuiltin(BuiltinOperator_SVDF, Register_SVDF());
    AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), 1, 3);
    AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION(), 1, 3);
    AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(), 1,
              3);
    AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D());
    AddBuiltin(BuiltinOperator_ABS, Register_ABS());
    AddBuiltin(BuiltinOperator_SIN, Register_SIN());
    AddBuiltin(BuiltinOperator_COS, Register_COS());
    AddBuiltin(BuiltinOperator_LOG, Register_LOG());
    AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
    AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT());
    AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());
    AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
    AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());
    AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM());
    AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM());
    AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX());
    AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN());
    AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
    AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
    AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
    AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
    AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL());
    AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL());
    AddBuiltin(BuiltinOperator_GREATER, Register_GREATER());
    AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL());
    AddBuiltin(BuiltinOperator_LESS, Register_LESS());
    AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL());
    AddBuiltin(BuiltinOperator_CEIL, Register_CEIL());
    AddBuiltin(BuiltinOperator_ROUND, Register_ROUND());
    AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE());
    AddBuiltin(BuiltinOperator_PACK, Register_PACK());
    AddBuiltin(BuiltinOperator_PAD, Register_PAD());
    AddBuiltin(BuiltinOperator_PADV2, Register_PADV2());
    AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(), 1, 3);
    AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK());
    AddBuiltin(BuiltinOperator_NEG, Register_NEG());
    AddBuiltin(BuiltinOperator_ADD, Register_ADD());
    AddBuiltin(BuiltinOperator_MUL, Register_MUL());
    AddBuiltin(BuiltinOperator_QUANTIZE, Register_QUANTIZE());
    AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE(), 1, 2);
    AddBuiltin(BuiltinOperator_RELU, Register_RELU());
    AddBuiltin(BuiltinOperator_RELU6, Register_RELU6());

    AddCustom("XC_argmax_16", Register_ArgMax_16());
    AddCustom("XC_maxpool2d_deep", Register_MaxPool());
    AddCustom("XC_avgpool2d_deep", Register_AvgPool());
    AddCustom("XC_fc_deepin_anyout", Register_FullyConnected_DIAO());
    AddCustom("XC_conv2d_shallowin_deepout_relu", Register_Conv_SIDO());
    AddCustom("XC_conv2d_deepin_deepout_relu", Register_Conv_DIDO());
    AddCustom("XC_requantize_16_to_8", Register_Requantize_16_to_8());
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite