//===- OptimizeLDSTest.cpp - Tests for OptimizeLDSUtility -----------------===//

#include "third_party/amd/lib/TritonAMDGPUToLLVM/OptimizeLDSUtility.h"
#include <gtest/gtest.h>
#include <numeric>

namespace mlir {

template <unsigned P> bool checkProdEq(ArrayRef<unsigned> a) {
  unsigned prod =
      std::reduce(a.begin(), a.end(), 1u, std::multiplies<unsigned>());
  return prod == P;
}

TEST(OptimizeLDSUtility, factorizePowerOf2) {
  int numwarps;
  int rank;
  // check rank=1 generation
  numwarps = 4;
  rank = 1;
  auto output1 = triton::AMD::factorizePowerOf2(numwarps, rank);
  ASSERT_EQ(output1.size(), 1);
  ASSERT_EQ(output1[0][0], numwarps);
  // check rank=2 generation
  numwarps = 8;
  rank = 2;
  auto output2 = triton::AMD::factorizePowerOf2(numwarps, rank);
  ASSERT_EQ(output2.size(), 4);
  ASSERT_TRUE(std::all_of(output2.begin(), output2.end(), checkProdEq<8>));
  ASSERT_TRUE(std::all_of(output2.begin(), output2.end(),
                          [](auto a) { return a.size() == 2; }));
  // check rank=3 generation
  numwarps = 8;
  rank = 3;
  auto output3 = triton::AMD::factorizePowerOf2(numwarps, rank);
  ASSERT_EQ(output3.size(), 10);
  ASSERT_TRUE(std::all_of(output3.begin(), output3.end(), checkProdEq<8>));
  ASSERT_TRUE(std::all_of(output3.begin(), output3.end(),
                          [](auto a) { return a.size() == 3; }));
}

} // namespace mlir
