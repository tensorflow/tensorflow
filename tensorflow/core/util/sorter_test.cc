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

#include "tensorflow/core/util/sorter.h"
#include <random>
#include <vector>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

template <typename T>
struct Less {
  inline bool operator()(const T i, const T j) const { return i < j; }
};

template <typename T>
struct Greater {
  inline bool operator()(const T i, const T j) const { return i > j; }
};

void GenRandomUInt64(int N, std::vector<uint64>* vec) {
  random::PhiloxRandom philox(288, 19);
  random::SimplePhilox rnd(&philox);
  vec->resize(N);
  for (int i = 0; i < N; ++i) {
    (*vec)[i] = rnd.Rand64();
  }
}

void GenRandomFloat(int N, std::vector<float>* vec) {
  random::PhiloxRandom philox(392, 44);
  random::SimplePhilox rnd(&philox);
  vec->resize(N);
  for (int i = 0; i < N; ++i) {
    (*vec)[i] = rnd.RandFloat();
  }
}

TEST(SorterQSort, UInt64Less) {
  thread::ThreadPool threads(Env::Default(), "test", 16);
  Less<uint64> less;
  for (int num : {0, 1, 100, 1000, 10000, 100000}) {
    for (int nth : {-1, 0, 4, 16}) {
      std::vector<uint64> a, b;
      GenRandomUInt64(num, &a);
      b = a;
      ParallelSorter sorter(nth, &threads);
      sorter.QSort(a, less);
      std::sort(b.begin(), b.end(), less);
      EXPECT_EQ(a, b);
    }
  }
}

TEST(SorterQSort, FloatGreater) {
  thread::ThreadPool threads(Env::Default(), "test", 16);
  Greater<float> greater;
  for (int num : {0, 1, 100, 1000, 10000, 100000}) {
    for (int nth : {-1, 0, 4, 16}) {
      std::vector<float> a, b;
      GenRandomFloat(num, &a);
      b = a;
      ParallelSorter sorter(nth, &threads);
      sorter.QSort(a, greater);
      std::sort(b.begin(), b.end(), greater);
      EXPECT_EQ(a, b);
    }
  }
}

}  // namespace
}  // namespace tensorflow
