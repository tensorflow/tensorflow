/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_CODER_KERNELS_RANGE_CODER_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_CODER_KERNELS_RANGE_CODER_H_

#include <limits>
#include <string>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class RangeEncoder {
 public:
  // `precision` determines the granularity of probability masses passed to
  // Encode() function below.
  //
  // REQUIRES: 0 < precision <= 16.
  explicit RangeEncoder(int precision);

  // Encodes a half-open interval [lower / 2^precision, upper / 2^precision).
  // Suppose each character to be encoded is from an integer-valued
  // distribution. When encoding a random character x0, the arguments lower and
  // upper represent
  //   Pr(X < x0) = lower / 2^precision,
  //   Pr(X < x0 + 1) = upper / 2^precision,
  // where X is a random variable following the distribution.
  //
  // For example, assume that the distribution has possible outputs 0, 1, 2, ...
  // To encode value 0, lower = 0 and upper = Pr(X = 0).
  // To encode value 1, lower = Pr(X = 0) and upper = Pr(X = 0 or 1).
  // To encode value 2, lower = Pr(X = 0 or 1) and upper = Pr(X = 0, 1, or 2).
  // ...
  //
  // REQUIRES: 0 <= lower < upper <= 2^precision.
  void Encode(int32 lower, int32 upper, string* sink);

  // The encode may contain some under-determined values from previous encoding.
  // After Encode() calls, Finalize() must be called. Otherwise the encoded
  // string may not be decoded.
  void Finalize(string* sink);

 private:
  uint32 base_ = 0;
  uint32 size_minus1_ = std::numeric_limits<uint32>::max();
  uint64 delay_ = 0;

  const int precision_;
};

class RangeDecoder {
 public:
  // Holds a reference to `source`. The caller has to make sure that `source`
  // outlives the decoder object.
  //
  // REQUIRES: `precision` must be the same as the encoder's precision.
  // REQUIRES: 0 < precision <= 16.
  RangeDecoder(const string& source, int precision);

  // Decodes a character from `source` using CDF. The size of `cdf` should be
  // one more than the number of the character in the alphabet.
  //
  // If x0, x1, x2, ... are the possible characters (in increasing order) from
  // the distribution, then
  //   cdf[0] = 0
  //   cdf[1] = Pr(X <= x0),
  //   cdf[2] = Pr(X <= x1),
  //   cdf[3] = Pr(X <= x2),
  //   ...
  //
  // The returned value is an index to `cdf` where the decoded character
  // corresponds to.
  //
  // REQUIRES: cdf.size() > 1.
  // REQUIRES: cdf[i] <= cdf[i + 1] for i = 0, 1, ..., cdf.size() - 2.
  // REQUIRES: cdf[cdf.size() - 1] <= 2^precision.
  //
  // In practice the last element of `cdf` should equal to 2^precision.
  int32 Decode(gtl::ArraySlice<int32> cdf);

 private:
  void Read16BitValue();

  uint32 base_ = 0;
  uint32 size_minus1_ = std::numeric_limits<uint32>::max();
  uint32 value_ = 0;

  string::const_iterator current_;
  const string::const_iterator begin_;
  const string::const_iterator end_;

  const int precision_;
};
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_CODER_KERNELS_RANGE_CODER_H_
