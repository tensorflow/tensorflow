#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("AttentionMask")
    .Attr("fill_value: float")
    .Input("sequence_len: int64")
    .Input("input: float")
    .Output("output: float")
    .Doc(R"doc(
AttentionMask
)doc");

REGISTER_OP("AttentionMaskMedian")
    .Attr("fill_value: float")
    .Attr("window_l: int = 10")
    .Attr("window_r: int = 200")
    .Input("sequence_len: int64")
    .Input("input: float")
    .Input("prev_alignment: float")
    .Output("output: float")
    .Doc(R"doc(
AttentionMaskMedian
)doc");

}  // end namespace tensorflow
