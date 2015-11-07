#include "tensorflow/core/graph/tensor_id.h"
#include <gtest/gtest.h>
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

static string ParseHelper(const string& n) {
  TensorId id = ParseTensorName(n);
  return strings::StrCat(id.first, ":", id.second);
}

TEST(TensorIdTest, ParseTensorName) {
  EXPECT_EQ(ParseHelper("W1"), "W1:0");
  EXPECT_EQ(ParseHelper("weights:0"), "weights:0");
  EXPECT_EQ(ParseHelper("W1:1"), "W1:1");
  EXPECT_EQ(ParseHelper("W1:17"), "W1:17");
  EXPECT_EQ(ParseHelper("xyz1_17"), "xyz1_17:0");
}

static uint32 Skewed(random::SimplePhilox* rnd, int max_log) {
  const uint32 space = 1 << (rnd->Rand32() % (max_log + 1));
  return rnd->Rand32() % space;
}

static void BM_ParseTensorName(int iters, int arg) {
  testing::StopTiming();
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<string> names;
  for (int i = 0; i < 100; i++) {
    string name;
    switch (arg) {
      case 0: {  // Generate random names
        size_t len = Skewed(&rnd, 4);
        while (name.size() < len) {
          name += rnd.OneIn(4) ? '0' : 'a';
        }
        if (rnd.OneIn(3)) {
          strings::StrAppend(&name, ":", rnd.Uniform(12));
        }
        break;
      }
      case 1:
        name = "W1";
        break;
      case 2:
        name = "t0003";
        break;
      case 3:
        name = "weights";
        break;
      case 4:
        name = "weights:17";
        break;
      default:
        LOG(FATAL) << "Unexpected arg";
        break;
    }
    names.push_back(name);
  }
  testing::StartTiming();
  TensorId id;
  int index = 0;
  int sum = 0;
  while (--iters > 0) {
    id = ParseTensorName(names[index++ % names.size()]);
    sum += id.second;
  }
  VLOG(2) << sum;  // Prevent compiler from eliminating loop body
}
BENCHMARK(BM_ParseTensorName)->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Arg(4);

}  // namespace
}  // namespace tensorflow
