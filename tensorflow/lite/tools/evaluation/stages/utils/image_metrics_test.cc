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
#include "tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h"

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace evaluation {
namespace image {

using testing::_;
using testing::Eq;
using testing::FloatEq;
using testing::FloatNear;

// Find the max precision with the minimum recall.
float MaxP(float minr, const std::vector<PR>& prs) {
  float p = 0;
  for (auto& pr : prs) {
    if (pr.r >= minr) p = std::max(p, pr.p);
  }
  return p;
}

float ExpectedAP(const std::vector<PR>& prs) {
  float sum = 0;
  for (float r = 0; r <= 1.0; r += 0.01) {
    sum += MaxP(r, prs);
  }
  return sum / 101;
}

float GenerateRandomFraction() {
  return static_cast<float>(std::rand()) / RAND_MAX;
}

TEST(ImageMetricsTest, APBasic) {
  std::vector<PR> prs;

  prs = {{1., 1.}, {0.5, 1.0}, {1 / 3, 1.0}};
  EXPECT_NEAR(ExpectedAP(prs), AveragePrecision().FromPRCurve(prs), 1e-6);

  prs = {{1.0, 0.01}};
  EXPECT_NEAR(ExpectedAP(prs), AveragePrecision().FromPRCurve(prs), 1e-6);

  prs = {{1.0, 0.2}, {1.0, 0.4},  {0.67, 0.4}, {0.5, 0.4},  {0.4, 0.4},
         {0.5, 0.6}, {0.57, 0.8}, {0.5, 0.8},  {0.44, 0.8}, {0.5, 1.0}};
  EXPECT_NEAR(ExpectedAP(prs), AveragePrecision().FromPRCurve(prs), 1e-6);
}

TEST(ImageMetricsTest, APRandom) {
  // Generates a set of p-r points.
  std::vector<PR> prs;
  for (int i = 0; i < 5000; ++i) {
    float p = GenerateRandomFraction();
    float r = GenerateRandomFraction();
    prs.push_back({p, r});
  }

  const float expected = ExpectedAP(prs);

  // Sort them w/ recall non-decreasing.
  std::sort(std::begin(prs), std::end(prs),
            [](const PR& a, const PR& b) { return a.r < b.r; });
  const float actual = AveragePrecision().FromPRCurve(prs);

  EXPECT_NEAR(expected, actual, 1e-5);
}

TEST(ImageMetricsTest, BBoxAPBasic) {
  std::vector<Detection> gt;
  gt.push_back(Detection({false, 100, 1, {{0, 1}, {0, 1}}}));
  gt.push_back(Detection({false, 200, 1, {{1, 2}, {1, 2}}}));
  std::vector<Detection> pd;
  pd.push_back(Detection({false, 100, 0.8, {{0.1, 1.1}, {0.1, 1.1}}}));
  pd.push_back(Detection({false, 200, 0.8, {{0.9, 1.9}, {0.9, 1.9}}}));
  EXPECT_NEAR(1.0, AveragePrecision().FromBoxes(gt, pd), 1e-6);
  AveragePrecision::Options opts;
  opts.iou_threshold = 0.85;
  EXPECT_NEAR(0.0, AveragePrecision(opts).FromBoxes(gt, pd), 1e-6);
}

TEST(ImageMetricsTest, Box2DOverlap) {
  Box2D a({{0, 1}, {0, 1}});
  Box2D b({{0.5, 2.5}, {0.5, 2.5}});
  // Upper right quarter of box a overlaps.
  EXPECT_NEAR(0.25, a.Overlap(b), 1e-6);

  // Not symmetric if a and b have different areas.  Only lower left 0.5
  // of b overlaps, so a total of 0.25 over an area of 4 overlaps.
  EXPECT_NEAR(0.0625, b.Overlap(a), 1e-6);
}

TEST(ImageMetricsTest, BBoxAPwithIgnoredGroundTruth) {
  std::vector<Detection> gt;
  std::vector<Detection> pd;
  gt.push_back(Detection({false, 100, 1, {{1, 2}, {1, 2}}, kIgnoreOneMatch}));
  pd.push_back(Detection({false, 100, 0.8, {{0.1, 1.1}, {0.1, 1.1}}}));
  // All gt box are ignored, expect NaN.
  EXPECT_TRUE(std::isnan(AveragePrecision().FromBoxes(gt, pd)));

  gt.push_back({false, 100, 1, {{0, 1}, {0, 1}}});
  // Two gt and one pd, ap=1 because the unmatched gt is ignored.
  EXPECT_NEAR(1.0, AveragePrecision().FromBoxes(gt, pd), 1e-6);

  // Two gt and two pd, ap=1.
  pd.push_back({false, 100, 0.9, {{0.9, 1.9}, {0.9, 1.9}}});
  EXPECT_NEAR(1.0, AveragePrecision().FromBoxes(gt, pd), 1e-6);

  pd.push_back({false, 100, 0.95, {{0.9, 1.9}, {0.9, 1.9}}});

  // Two gt and three pd, one pair get ignored. So it's actually one gt with
  // two pd.
  EXPECT_NEAR(0.5, AveragePrecision().FromBoxes(gt, pd), 1e-6);
  gt[0].ignore = kIgnoreAllMatches;
  EXPECT_NEAR(1.0, AveragePrecision().FromBoxes(gt, pd), 1e-6);
}

TEST(ImageMetricsTest, BBoxAPRandom) {
  auto rand = [](int64_t id) {
    auto xmin = GenerateRandomFraction();
    auto xmax = xmin + GenerateRandomFraction();
    auto ymin = GenerateRandomFraction();
    auto ymax = ymin + GenerateRandomFraction();
    return Detection(
        {false, id, GenerateRandomFraction(), {{xmin, xmax}, {ymin, ymax}}});
  };
  std::vector<Detection> gt;
  for (int i = 0; i < 100; ++i) {
    gt.push_back(rand(i % 10));
  }
  std::vector<Detection> pd = gt;
  for (int i = 0; i < 10000; ++i) {
    pd.push_back(rand(i % 10));
  }
  std::vector<PR> pr;
  // Test pr curve output.
  AveragePrecision().FromBoxes(gt, pd, &pr);
  // Default num_recall_points=100, so there are p-r pairs of 101 levels.
  EXPECT_EQ(101, pr.size());
}

}  // namespace image
}  // namespace evaluation
}  // namespace tflite
