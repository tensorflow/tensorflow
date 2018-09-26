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

#ifndef TENSORFLOW_UTIL_SORTER_H_
#define TENSORFLOW_UTIL_SORTER_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

int divup(int a, int b) { return (a + (b - 1)) / b; }

// Recursively re-order x, which makes all elements within [first, pos) less
// than those in [pos, last).
template <typename T, class Compare>
void DivideAtPos(std::vector<T>& x, Compare& comp, int first, int last,
                 int pos) {
  if (first > pos || pos >= last) return;
  VLOG(2) << "[Sorter] Divide " << first << " to " << last << " at " << pos;

  int i = first, j = last - 1;
  T pivot = x[pos];
  while (true) {
    while (i <= j && comp(x[i], pivot)) ++i;
    while (i <= j && !comp(x[j], pivot)) --j;
    if (i >= j) break;
    std::swap(x[i], x[j]);
    ++i;
    --j;
  }
  if (i >= last) {
    i = last - 1;
    std::swap(x[pos], x[i]);
  }
  if (j < first) {
    std::swap(x[pos], x[first]);
    i = first + 1;
  }
  if (i == pos) return;
  if (i > pos) {
    DivideAtPos(x, comp, first, i, pos);
  } else {
    DivideAtPos(x, comp, i, last, pos);
  }
}

}  // namespace

// Parallel sorter class for sorting a set of elements with type of 'T' with
// the comparing logic provided by 'Compare'. The sorter will use the thread
// pool to parallelly complete the sorting of all elements.
//
// Member variables:
//   'nthreads_' > 0: the number of total sorting threads would be used.
//   'threadpool_': the pool to schedule the sorting threads, if it is nullptr,
//                  serial execution would be performed.
class ParallelSorter {
 public:
  ParallelSorter() { ParallelSorter(1, nullptr); }

  ParallelSorter(int nthreads, thread::ThreadPool* threadpool)
      : nthreads_(nthreads), threadpool_(threadpool) {
    if (nthreads <= 1 || threadpool == nullptr) {
      nthreads_ = 1;
      threadpool_ = nullptr;
    }
  }

  // Apply the sorting with qsort based algorithm.
  template <typename T, class Compare>
  void QSort(std::vector<T>& x, const Compare& comp) {
    if (x.size() <= 1) return;
    if (nthreads_ <= 1 || threadpool_ == nullptr) {
      std::sort(x.begin(), x.end(), comp);
      return;
    }

#define HALF_L1_CACHE_SIZE 16384
    const int block_size = HALF_L1_CACHE_SIZE / sizeof(T);
    const int block_count = divup(x.size(), block_size);
#undef HALF_L1_CACHE_SIZE
    BlockingCounter counter(block_count);
    std::function<void(int, int)> SortRange;
    SortRange = [this, &x, &comp, block_size, &counter, &SortRange](int first,
                                                                    int last) {
      VLOG(2) << "[Parallel QSort] Sorting " << first << " to " << last;
      // Single block or less, execute directly.
      if (last - first <= block_size) {
        std::sort(x.begin() + first, x.begin() + last, comp);
        counter.DecrementCount();
        return;
      }
      // Split into blocs and submit to the pool.
      int mid = first + divup((last - first) / 2, block_size) * block_size;
      DivideAtPos(x, comp, first, last, mid);
      threadpool_->Schedule([=, &SortRange]() { SortRange(mid, last); });
      SortRange(first, mid);
    };
    SortRange(0, x.size());
    counter.Wait();
  }

 private:
  int nthreads_;
  thread::ThreadPool* threadpool_;
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_SORTER_H_
