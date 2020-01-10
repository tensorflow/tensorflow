#include "tensorflow/core/lib/histogram/histogram.h"
#include <float.h>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/summary.pb.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace histogram {

static void Validate(const Histogram& h) {
  string s1 = h.ToString();
  LOG(ERROR) << s1;

  HistogramProto proto_with_zeroes;
  h.EncodeToProto(&proto_with_zeroes, true);
  Histogram h2;
  EXPECT_TRUE(h2.DecodeFromProto(proto_with_zeroes));
  string s2 = h2.ToString();
  LOG(ERROR) << s2;

  EXPECT_EQ(s1, s2);

  HistogramProto proto_no_zeroes;
  h.EncodeToProto(&proto_no_zeroes, false);
  LOG(ERROR) << proto_no_zeroes.DebugString();
  Histogram h3;
  EXPECT_TRUE(h3.DecodeFromProto(proto_no_zeroes));
  string s3 = h3.ToString();
  LOG(ERROR) << s3;

  EXPECT_EQ(s1, s3);
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

TEST(Histogram, Percentile) {
  Histogram h({0, 10, 100, DBL_MAX});
  h.Add(-2);
  h.Add(-2);
  h.Add(0);
  double median = h.Percentile(50.0);
  EXPECT_EQ(median, -0.5);
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
}  // namespace tensorflow
