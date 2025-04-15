/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/histogram/histogram.h"

#include <float.h>

#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/histogram.pb.h"

namespace tsl {
namespace histogram {

static void Validate(const Histogram& h) {
  HistogramProto proto_with_zeroes;
  h.EncodeToProto(&proto_with_zeroes, true);
  Histogram h2;
  EXPECT_TRUE(h2.DecodeFromProto(proto_with_zeroes));

  EXPECT_EQ(h2.ToString(), h.ToString());

  HistogramProto proto_no_zeroes;
  h.EncodeToProto(&proto_no_zeroes, false);
  Histogram h3;
  EXPECT_TRUE(h3.DecodeFromProto(proto_no_zeroes));
  string s3 = h3.ToString();
  LOG(ERROR) << s3;

  EXPECT_EQ(h3.ToString(), h.ToString());
}

TEST(Histogram, Empty) {
  Histogram h;
  Validate(h);
}

TEST(Histogram, SingleValue) {
  Histogram h;
  h.Add(-3.0);
  Validate(h);
}

TEST(Histogram, CustomBuckets) {
  Histogram h({-10, -5, 0, 5, 10, 100, 1000, 10000, DBL_MAX});
  h.Add(-3.0);
  h.Add(4.99);
  h.Add(5.0);
  h.Add(1000.0);
  Validate(h);
}

TEST(Histogram, Median) {
  Histogram h({0, 10, 100, DBL_MAX});
  h.Add(-2);
  h.Add(-2);
  h.Add(0);
  double median = h.Median();
  EXPECT_EQ(median, -0.5);
}

TEST(Histogram, Percentile) {
  // 10%, 30%, 40%, 20%
  Histogram h({1, 2, 3, 4});
  // 10% first bucket
  h.Add(-1.0);
  // 30% second bucket
  h.Add(1.5);
  h.Add(1.5);
  h.Add(1.5);
  // 40% third bucket
  h.Add(2.5);
  h.Add(2.5);
  h.Add(2.5);
  h.Add(2.5);
  // 20% fourth bucket
  h.Add(3.5);
  h.Add(3.9);

  EXPECT_EQ(h.Percentile(0), -1.0);    // -1.0 = histo.min_
  EXPECT_EQ(h.Percentile(25), 1.5);    // 1.5 = remap(25, 10, 40, 1, 2)
  EXPECT_EQ(h.Percentile(50), 2.25);   // 2.25 = remap(50, 40, 80, 2, 3)
  EXPECT_EQ(h.Percentile(75), 2.875);  // 2.875 = remap(75, 40, 80, 2, 3)
  EXPECT_EQ(h.Percentile(90), 3.45);   // 3.45 = remap(90, 80, 100, 3, 3.9)
  EXPECT_EQ(h.Percentile(100), 3.9);   // 3.9 = histo.max_
}

TEST(Histogram, Basic) {
  Histogram h;
  for (int i = 0; i < 100; i++) {
    h.Add(i);
  }
  for (int i = 1000; i < 100000; i += 1000) {
    h.Add(i);
  }
  Validate(h);
}

TEST(ThreadSafeHistogram, Basic) {
  // Fill a normal histogram.
  Histogram h;
  for (int i = 0; i < 100; i++) {
    h.Add(i);
  }

  // Fill a thread-safe histogram with the same values.
  ThreadSafeHistogram tsh;
  for (int i = 0; i < 100; i++) {
    tsh.Add(i);
  }

  for (int i = 0; i < 2; ++i) {
    bool preserve_zero_buckets = (i == 0);
    HistogramProto h_proto;
    h.EncodeToProto(&h_proto, preserve_zero_buckets);
    HistogramProto tsh_proto;
    tsh.EncodeToProto(&tsh_proto, preserve_zero_buckets);

    // Let's decode from the proto of the other histogram type.
    Histogram h2;
    EXPECT_TRUE(h2.DecodeFromProto(tsh_proto));
    ThreadSafeHistogram tsh2;
    EXPECT_TRUE(tsh2.DecodeFromProto(h_proto));

    // Now let's reencode and check they match.
    EXPECT_EQ(h2.ToString(), tsh2.ToString());
  }

  EXPECT_EQ(h.Median(), tsh.Median());
  EXPECT_EQ(h.Percentile(40.0), tsh.Percentile(40.0));
  EXPECT_EQ(h.Average(), tsh.Average());
  EXPECT_EQ(h.StandardDeviation(), tsh.StandardDeviation());
  EXPECT_EQ(h.ToString(), tsh.ToString());
}

}  // namespace histogram
}  // namespace tsl
