#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("BernoulliSample")
    .Input("p: float")
    .Input("a: T")
    .Input("b: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
Sample `a` with probability `p`, or equivalently sample `b` with probability
(1 - `p`).

p: The probability to sample `a`.
a: Heads.
b: Tails.
output: The sampled results which is either `a` or `b`.
)doc");

REGISTER_OP("SampleDistributionIndex")
    .Input("p: float")
    .Output("output: int64")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Doc(R"doc(
Sample from distribution `p` (i.e., sum(p) = 1) and return the index.

p: The probability distribution to sample from.
output: The index of the sample.
)doc");

}  // end namespace tensorflow
