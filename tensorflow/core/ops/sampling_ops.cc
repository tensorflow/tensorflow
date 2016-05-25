/* Copyright 2016 Google Inc. All Rights Reserved.

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
