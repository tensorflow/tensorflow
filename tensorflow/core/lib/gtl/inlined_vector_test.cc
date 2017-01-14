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

#include "tensorflow/core/lib/gtl/inlined_vector.h"

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef tensorflow::gtl::InlinedVector<int, 8> IntVec;

// A type that counts number of live occurrences of the type
static int64 instances = 0;
class Instance {
 public:
  int value_;
  explicit Instance(int x) : value_(x) { instances++; }
  Instance(const Instance& x) : value_(x.value_) { instances++; }
  ~Instance() { instances--; }

  friend inline void swap(Instance& a, Instance& b) {
    using std::swap;
    swap(a.value_, b.value_);
  }

  friend std::ostream& operator<<(std::ostream& o, const Instance& v) {
    return o << "[value:" << v.value_ << "]";
  }
};

typedef tensorflow::gtl::InlinedVector<Instance, 8> InstanceVec;

// A simple reference counted class to make sure that the proper elements are
// destroyed in the erase(begin, end) test.
class RefCounted {
 public:
  RefCounted(int value, int* count) : value_(value), count_(count) { Ref(); }

  RefCounted(const RefCounted& v) : value_(v.value_), count_(v.count_) {
    VLOG(5) << "[RefCounted: copy"
            << " from count @" << v.count_ << "]";
    Ref();
  }

  ~RefCounted() {
    Unref();
    count_ = NULL;
  }

  friend void swap(RefCounted& a, RefCounted& b) {
    using std::swap;
    swap(a.value_, b.value_);
    swap(a.count_, b.count_);
  }

  RefCounted& operator=(RefCounted v) {
    using std::swap;
    swap(*this, v);
    return *this;
  }

  void Ref() const {
    CHECK(count_ != NULL);
    ++(*count_);
    VLOG(5) << "[Ref: refcount " << *count_ << " on count @" << count_ << "]";
  }

  void Unref() const {
    --(*count_);
    CHECK_GE(*count_, 0);
    VLOG(5) << "[Unref: refcount " << *count_ << " on count @" << count_ << "]";
  }

  int count() const { return *count_; }

  friend std::ostream& operator<<(std::ostream& o, const RefCounted& v) {
    return o << "[value:" << v.value_ << ", count:" << *v.count_ << "]";
  }

  int value_;
  int* count_;
};

typedef tensorflow::gtl::InlinedVector<RefCounted, 8> RefCountedVec;

// A class with a vtable pointer
class Dynamic {
 public:
  virtual ~Dynamic() {}

  friend std::ostream& operator<<(std::ostream& o, const Dynamic& v) {
    return o << "[Dynamic]";
  }
};

typedef tensorflow::gtl::InlinedVector<Dynamic, 8> DynamicVec;

// Append 0..len-1 to *v
static void Fill(IntVec* v, int len, int offset = 0) {
  for (int i = 0; i < len; i++) {
    v->push_back(i + offset);
  }
}

static IntVec Fill(int len, int offset = 0) {
  IntVec v;
  Fill(&v, len, offset);
  return v;
}

TEST(IntVec, SimpleOps) {
  for (int len = 0; len < 20; len++) {
    IntVec v;
    const IntVec& cv = v;  // const alias

    Fill(&v, len);
    EXPECT_EQ(len, v.size());
    EXPECT_LE(len, v.capacity());

    for (int i = 0; i < len; i++) {
      EXPECT_EQ(i, v[i]);
    }
    EXPECT_EQ(v.begin(), v.data());
    EXPECT_EQ(cv.begin(), cv.data());

    int counter = 0;
    for (IntVec::iterator iter = v.begin(); iter != v.end(); ++iter) {
      EXPECT_EQ(counter, *iter);
      counter++;
    }
    EXPECT_EQ(counter, len);

    counter = 0;
    for (IntVec::const_iterator iter = v.begin(); iter != v.end(); ++iter) {
      EXPECT_EQ(counter, *iter);
      counter++;
    }
    EXPECT_EQ(counter, len);

    if (len > 0) {
      EXPECT_EQ(0, v.front());
      EXPECT_EQ(len - 1, v.back());
      v.pop_back();
      EXPECT_EQ(len - 1, v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(i, v[i]);
      }
    }
  }
}

TEST(IntVec, Erase) {
  for (int len = 1; len < 20; len++) {
    for (int i = 0; i < len; ++i) {
      IntVec v;
      Fill(&v, len);
      v.erase(v.begin() + i);
      EXPECT_EQ(len - 1, v.size());
      for (int j = 0; j < i; ++j) {
        EXPECT_EQ(j, v[j]);
      }
      for (int j = i; j < len - 1; ++j) {
        EXPECT_EQ(j + 1, v[j]);
      }
    }
  }
}

// At the end of this test loop, the elements between [erase_begin, erase_end)
// should have reference counts == 0, and all others elements should have
// reference counts == 1.
TEST(RefCountedVec, EraseBeginEnd) {
  for (int len = 1; len < 20; ++len) {
    for (int erase_begin = 0; erase_begin < len; ++erase_begin) {
      for (int erase_end = erase_begin; erase_end <= len; ++erase_end) {
        std::vector<int> counts(len, 0);
        RefCountedVec v;
        for (int i = 0; i < len; ++i) {
          v.push_back(RefCounted(i, &counts[i]));
        }

        int erase_len = erase_end - erase_begin;

        v.erase(v.begin() + erase_begin, v.begin() + erase_end);

        EXPECT_EQ(len - erase_len, v.size());

        // Check the elements before the first element erased.
        for (int i = 0; i < erase_begin; ++i) {
          EXPECT_EQ(i, v[i].value_);
        }

        // Check the elements after the first element erased.
        for (size_t i = erase_begin; i < v.size(); ++i) {
          EXPECT_EQ(i + erase_len, v[i].value_);
        }

        // Check that the elements at the beginning are preserved.
        for (int i = 0; i < erase_begin; ++i) {
          EXPECT_EQ(1, counts[i]);
        }

        // Check that the erased elements are destroyed
        for (int i = erase_begin; i < erase_end; ++i) {
          EXPECT_EQ(0, counts[i]);
        }

        // Check that the elements at the end are preserved.
        for (int i = erase_end; i < len; ++i) {
          EXPECT_EQ(1, counts[i]);
        }
      }
    }
  }
}

struct NoDefaultCtor {
  explicit NoDefaultCtor(int /* x */) {}
};
struct NoCopy {
  NoCopy() {}
  NoCopy(const NoCopy& /* x */) = delete;
};
struct NoAssign {
  NoAssign() {}
  NoAssign& operator=(const NoAssign& /* x */) = delete;
};
TEST(InlinedVectorTest, NoDefaultCtor) {
  tensorflow::gtl::InlinedVector<NoDefaultCtor, 1> v(10, NoDefaultCtor(2));
  (void)v;
}
TEST(InlinedVectorTest, NoCopy) {
  tensorflow::gtl::InlinedVector<NoCopy, 1> v(10);
  (void)v;
}
TEST(InlinedVectorTest, NoAssign) {
  tensorflow::gtl::InlinedVector<NoAssign, 1> v(10);
  (void)v;
}

TEST(IntVec, Insert) {
  for (int len = 0; len < 20; len++) {
    for (int pos = 0; pos <= len; pos++) {
      IntVec v;
      Fill(&v, len);
      v.insert(v.begin() + pos, 9999);
      EXPECT_EQ(v.size(), len + 1);
      for (int i = 0; i < pos; i++) {
        EXPECT_EQ(v[i], i);
      }
      EXPECT_EQ(v[pos], 9999);
      for (size_t i = pos + 1; i < v.size(); i++) {
        EXPECT_EQ(v[i], i - 1);
      }
    }
  }
}

TEST(RefCountedVec, InsertConstructorDestructor) {
  // Make sure the proper construction/destruction happen during insert
  // operations.
  for (int len = 0; len < 20; len++) {
    SCOPED_TRACE(len);
    for (int pos = 0; pos <= len; pos++) {
      SCOPED_TRACE(pos);
      std::vector<int> counts(len, 0);
      int inserted_count = 0;
      RefCountedVec v;
      for (int i = 0; i < len; ++i) {
        SCOPED_TRACE(i);
        v.push_back(RefCounted(i, &counts[i]));
      }

      for (auto elem : counts) {
        EXPECT_EQ(1, elem);
      }

      RefCounted insert_element(9999, &inserted_count);
      EXPECT_EQ(1, inserted_count);
      v.insert(v.begin() + pos, insert_element);
      EXPECT_EQ(2, inserted_count);
      // Check that the elements at the end are preserved.
      for (auto elem : counts) {
        EXPECT_EQ(1, elem);
      }
      EXPECT_EQ(2, inserted_count);
    }
  }
}

TEST(IntVec, Resize) {
  for (int len = 0; len < 20; len++) {
    IntVec v;
    Fill(&v, len);

    // Try resizing up and down by k elements
    static const int kResizeElem = 1000000;
    for (int k = 0; k < 10; k++) {
      // Enlarging resize
      v.resize(len + k, kResizeElem);
      EXPECT_EQ(len + k, v.size());
      EXPECT_LE(len + k, v.capacity());
      for (int i = 0; i < len + k; i++) {
        if (i < len) {
          EXPECT_EQ(i, v[i]);
        } else {
          EXPECT_EQ(kResizeElem, v[i]);
        }
      }

      // Shrinking resize
      v.resize(len, kResizeElem);
      EXPECT_EQ(len, v.size());
      EXPECT_LE(len, v.capacity());
      for (int i = 0; i < len; i++) {
        EXPECT_EQ(i, v[i]);
      }
    }
  }
}

TEST(IntVec, InitWithLength) {
  for (int len = 0; len < 20; len++) {
    IntVec v(len, 7);
    EXPECT_EQ(len, v.size());
    EXPECT_LE(len, v.capacity());
    for (int i = 0; i < len; i++) {
      EXPECT_EQ(7, v[i]);
    }
  }
}

TEST(IntVec, CopyConstructorAndAssignment) {
  for (int len = 0; len < 20; len++) {
    IntVec v;
    Fill(&v, len);
    EXPECT_EQ(len, v.size());
    EXPECT_LE(len, v.capacity());

    IntVec v2(v);
    EXPECT_EQ(v, v2);

    for (int start_len = 0; start_len < 20; start_len++) {
      IntVec v3;
      Fill(&v3, start_len, 99);  // Add dummy elements that should go away
      v3 = v;
      EXPECT_EQ(v, v3);
    }
  }
}

TEST(OverheadTest, Storage) {
  // Check for size overhead.
  using tensorflow::gtl::InlinedVector;
  EXPECT_EQ(2 * sizeof(int*), sizeof(InlinedVector<int*, 1>));
  EXPECT_EQ(4 * sizeof(int*), sizeof(InlinedVector<int*, 2>));
  EXPECT_EQ(4 * sizeof(int*), sizeof(InlinedVector<int*, 3>));
  EXPECT_EQ(6 * sizeof(int*), sizeof(InlinedVector<int*, 4>));

  EXPECT_EQ(2 * sizeof(char*), sizeof(InlinedVector<char, 1>));
  EXPECT_EQ(2 * sizeof(char*), sizeof(InlinedVector<char, 2>));
  EXPECT_EQ(2 * sizeof(char*), sizeof(InlinedVector<char, 3>));
  EXPECT_EQ(2 * sizeof(char*),
            sizeof(InlinedVector<char, 2 * sizeof(char*) - 1>));
  EXPECT_EQ(4 * sizeof(char*), sizeof(InlinedVector<char, 2 * sizeof(char*)>));
}

TEST(IntVec, Clear) {
  for (int len = 0; len < 20; len++) {
    SCOPED_TRACE(len);
    IntVec v;
    Fill(&v, len);
    v.clear();
    EXPECT_EQ(0, v.size());
    EXPECT_EQ(v.begin(), v.end());
  }
}

TEST(IntVec, Reserve) {
  for (size_t len = 0; len < 20; len++) {
    IntVec v;
    Fill(&v, len);

    for (size_t newlen = 0; newlen < 100; newlen++) {
      const int* start_rep = v.data();
      v.reserve(newlen);
      const int* final_rep = v.data();
      if (newlen <= len) {
        EXPECT_EQ(start_rep, final_rep);
      }
      EXPECT_LE(newlen, v.capacity());

      // Filling up to newlen should not change rep
      while (v.size() < newlen) {
        v.push_back(0);
      }
      EXPECT_EQ(final_rep, v.data());
    }
  }
}

template <typename T>
static std::vector<typename T::value_type> Vec(const T& src) {
  std::vector<typename T::value_type> result;
  for (const auto& elem : src) {
    result.push_back(elem);
  }
  return result;
}

TEST(IntVec, SelfRefPushBack) {
  std::vector<string> std_v;
  tensorflow::gtl::InlinedVector<string, 4> v;
  const string s = "A very long string to ensure heap.";
  std_v.push_back(s);
  v.push_back(s);
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(std_v, Vec(v));

    v.push_back(v.back());
    std_v.push_back(std_v.back());
  }
  EXPECT_EQ(std_v, Vec(v));
}

TEST(IntVec, Swap) {
  for (int l1 = 0; l1 < 20; l1++) {
    SCOPED_TRACE(l1);
    for (int l2 = 0; l2 < 20; l2++) {
      SCOPED_TRACE(l2);
      IntVec a = Fill(l1, 0);
      IntVec b = Fill(l2, 100);
      {
        using std::swap;
        swap(a, b);
      }
      EXPECT_EQ(l1, b.size());
      EXPECT_EQ(l2, a.size());
      for (int i = 0; i < l1; i++) {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, b[i]);
      }
      for (int i = 0; i < l2; i++) {
        SCOPED_TRACE(i);
        EXPECT_EQ(100 + i, a[i]);
      }
    }
  }
}

TEST(InstanceVec, Swap) {
  for (int l1 = 0; l1 < 20; l1++) {
    for (int l2 = 0; l2 < 20; l2++) {
      InstanceVec a, b;
      for (int i = 0; i < l1; i++) a.push_back(Instance(i));
      for (int i = 0; i < l2; i++) b.push_back(Instance(100 + i));
      EXPECT_EQ(l1 + l2, instances);
      {
        using std::swap;
        swap(a, b);
      }
      EXPECT_EQ(l1 + l2, instances);
      EXPECT_EQ(l1, b.size());
      EXPECT_EQ(l2, a.size());
      for (int i = 0; i < l1; i++) {
        EXPECT_EQ(i, b[i].value_);
      }
      for (int i = 0; i < l2; i++) {
        EXPECT_EQ(100 + i, a[i].value_);
      }
    }
  }
}

TEST(IntVec, EqualAndNotEqual) {
  IntVec a, b;
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);

  a.push_back(3);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);

  b.push_back(3);
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);

  b.push_back(7);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);

  a.push_back(6);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);

  a.clear();
  b.clear();
  for (int i = 0; i < 100; i++) {
    a.push_back(i);
    b.push_back(i);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);

    b[i] = b[i] + 1;
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    b[i] = b[i] - 1;  // Back to before
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }
}

TEST(IntVec, RelationalOps) {
  IntVec a, b;
  EXPECT_FALSE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(b > a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b <= a);
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(b >= a);
  b.push_back(3);
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a > b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(b <= a);
  EXPECT_FALSE(a >= b);
  EXPECT_TRUE(b >= a);
}

TEST(InstanceVec, CountConstructorsDestructors) {
  const int start = instances;
  for (int len = 0; len < 20; len++) {
    InstanceVec v;
    for (int i = 0; i < len; i++) {
      v.push_back(Instance(i));
    }
    EXPECT_EQ(start + len, instances);

    {  // Copy constructor should create 'len' more instances.
      InstanceVec v_copy(v);
      EXPECT_EQ(start + len + len, instances);
    }
    EXPECT_EQ(start + len, instances);

    // Enlarging resize() must construct some objects
    v.resize(len + 10, Instance(100));
    EXPECT_EQ(start + len + 10, instances);

    // Shrinking resize() must destroy some objects
    v.resize(len, Instance(100));
    EXPECT_EQ(start + len, instances);

    // reserve() must not increase the number of initialized objects
    v.reserve(len + 1000);
    EXPECT_EQ(start + len, instances);

    // pop_back() and erase() must destroy one object
    if (len > 0) {
      v.pop_back();
      EXPECT_EQ(start + len - 1, instances);
      if (!v.empty()) {
        v.erase(v.begin());
        EXPECT_EQ(start + len - 2, instances);
      }
    }
  }
  EXPECT_EQ(start, instances);
}

TEST(InstanceVec, CountConstructorsDestructorsOnAssignment) {
  const int start = instances;
  for (int len = 0; len < 20; len++) {
    for (int longorshort = 0; longorshort <= 1; ++longorshort) {
      InstanceVec longer, shorter;
      for (int i = 0; i < len; i++) {
        longer.push_back(Instance(i));
        shorter.push_back(Instance(i));
      }
      longer.push_back(Instance(len));
      EXPECT_EQ(start + len + len + 1, instances);

      if (longorshort) {
        shorter = longer;
        EXPECT_EQ(start + (len + 1) + (len + 1), instances);
      } else {
        longer = shorter;
        EXPECT_EQ(start + len + len, instances);
      }
    }
  }
  EXPECT_EQ(start, instances);
}

TEST(RangedConstructor, SimpleType) {
  std::vector<int> source_v = {4, 5, 6, 7};
  // First try to fit in inline backing
  tensorflow::gtl::InlinedVector<int, 4> v(source_v.begin(), source_v.end());
  tensorflow::gtl::InlinedVector<int, 4> empty4;
  EXPECT_EQ(4, v.size());
  EXPECT_EQ(empty4.capacity(), v.capacity());  // Must still be inline
  EXPECT_EQ(4, v[0]);
  EXPECT_EQ(5, v[1]);
  EXPECT_EQ(6, v[2]);
  EXPECT_EQ(7, v[3]);

  // Now, force a re-allocate
  tensorflow::gtl::InlinedVector<int, 2> realloc_v(source_v.begin(),
                                                   source_v.end());
  tensorflow::gtl::InlinedVector<int, 2> empty2;
  EXPECT_EQ(4, realloc_v.size());
  EXPECT_LT(empty2.capacity(), realloc_v.capacity());
  EXPECT_EQ(4, realloc_v[0]);
  EXPECT_EQ(5, realloc_v[1]);
  EXPECT_EQ(6, realloc_v[2]);
  EXPECT_EQ(7, realloc_v[3]);
}

TEST(RangedConstructor, ComplexType) {
  // We also use a list here to pass a different flavor of iterator (e.g. not
  // random-access).
  std::list<Instance> source_v = {Instance(0)};

  // First try to fit in inline backing
  tensorflow::gtl::InlinedVector<Instance, 1> v(source_v.begin(),
                                                source_v.end());
  tensorflow::gtl::InlinedVector<Instance, 1> empty1;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(empty1.capacity(), v.capacity());  // Must still be inline
  EXPECT_EQ(0, v[0].value_);

  std::list<Instance> source_v2 = {Instance(0), Instance(1), Instance(2),
                                   Instance(3)};
  // Now, force a re-allocate
  tensorflow::gtl::InlinedVector<Instance, 1> realloc_v(source_v2.begin(),
                                                        source_v2.end());
  EXPECT_EQ(4, realloc_v.size());
  EXPECT_LT(empty1.capacity(), realloc_v.capacity());
  EXPECT_EQ(0, realloc_v[0].value_);
  EXPECT_EQ(1, realloc_v[1].value_);
  EXPECT_EQ(2, realloc_v[2].value_);
  EXPECT_EQ(3, realloc_v[3].value_);
}

TEST(RangedConstructor, ElementsAreConstructed) {
  std::vector<string> source_v = {"cat", "dog"};

  // Force expansion and re-allocation of v.  Ensures that when the vector is
  // expanded that new elements are constructed.
  tensorflow::gtl::InlinedVector<string, 1> v(source_v.begin(), source_v.end());
  EXPECT_EQ("cat", v[0]);
  EXPECT_EQ("dog", v[1]);
}

TEST(InitializerListConstructor, SimpleTypeWithInlineBacking) {
  auto vec = tensorflow::gtl::InlinedVector<int, 3>{4, 5, 6};
  EXPECT_EQ(3, vec.size());
  EXPECT_EQ(3, vec.capacity());
  EXPECT_EQ(4, vec[0]);
  EXPECT_EQ(5, vec[1]);
  EXPECT_EQ(6, vec[2]);
}

TEST(InitializerListConstructor, SimpleTypeWithReallocationRequired) {
  auto vec = tensorflow::gtl::InlinedVector<int, 2>{4, 5, 6};
  EXPECT_EQ(3, vec.size());
  EXPECT_LE(3, vec.capacity());
  EXPECT_EQ(4, vec[0]);
  EXPECT_EQ(5, vec[1]);
  EXPECT_EQ(6, vec[2]);
}

TEST(InitializerListConstructor, DisparateTypesInList) {
  EXPECT_EQ((std::vector<int>{-7, 8}),
            Vec(tensorflow::gtl::InlinedVector<int, 2>{-7, 8ULL}));

  EXPECT_EQ(
      (std::vector<string>{"foo", "bar"}),
      Vec(tensorflow::gtl::InlinedVector<string, 2>{"foo", string("bar")}));
}

TEST(InitializerListConstructor, ComplexTypeWithInlineBacking) {
  tensorflow::gtl::InlinedVector<Instance, 1> empty;
  auto vec = tensorflow::gtl::InlinedVector<Instance, 1>{Instance(0)};
  EXPECT_EQ(1, vec.size());
  EXPECT_EQ(empty.capacity(), vec.capacity());
  EXPECT_EQ(0, vec[0].value_);
}

TEST(InitializerListConstructor, ComplexTypeWithReallocationRequired) {
  auto vec =
      tensorflow::gtl::InlinedVector<Instance, 1>{Instance(0), Instance(1)};
  EXPECT_EQ(2, vec.size());
  EXPECT_LE(2, vec.capacity());
  EXPECT_EQ(0, vec[0].value_);
  EXPECT_EQ(1, vec[1].value_);
}

TEST(DynamicVec, DynamicVecCompiles) {
  DynamicVec v;
  (void)v;
}

static void BM_InlinedVectorFill(int iters, int len) {
  for (int i = 0; i < iters; i++) {
    IntVec v;
    for (int j = 0; j < len; j++) {
      v.push_back(j);
    }
  }
  testing::BytesProcessed((static_cast<int64>(iters) * len) * sizeof(int));
}
BENCHMARK(BM_InlinedVectorFill)->Range(0, 1024);

static void BM_InlinedVectorFillRange(int iters, int len) {
  std::unique_ptr<int[]> ia(new int[len]);
  for (int j = 0; j < len; j++) {
    ia[j] = j;
  }
  for (int i = 0; i < iters; i++) {
    IntVec TF_ATTRIBUTE_UNUSED v(ia.get(), ia.get() + len);
  }
  testing::BytesProcessed((static_cast<int64>(iters) * len) * sizeof(int));
}
BENCHMARK(BM_InlinedVectorFillRange)->Range(0, 1024);

static void BM_StdVectorFill(int iters, int len) {
  for (int i = 0; i < iters; i++) {
    std::vector<int> v;
    for (int j = 0; j < len; j++) {
      v.push_back(j);
    }
  }
  testing::BytesProcessed((static_cast<int64>(iters) * len) * sizeof(int));
}
BENCHMARK(BM_StdVectorFill)->Range(0, 1024);

namespace {
struct Buffer {  // some arbitrary structure for benchmarking.
  char* base;
  int length;
  int capacity;
  void* user_data;
};
}  // anonymous namespace

static void BM_InlinedVectorTenAssignments(int iters, int len) {
  typedef tensorflow::gtl::InlinedVector<Buffer, 2> BufferVec;

  BufferVec src;
  src.resize(len);

  iters *= 10;
  BufferVec dst;
  for (int i = 0; i < iters; i++) {
    dst = src;
  }
}
BENCHMARK(BM_InlinedVectorTenAssignments)
    ->Arg(0)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(4)
    ->Arg(20);

static void BM_CreateFromInitializerList(int iters) {
  for (; iters > 0; iters--) {
    tensorflow::gtl::InlinedVector<int, 4> x{1, 2, 3};
    (void)x[0];
  }
}
BENCHMARK(BM_CreateFromInitializerList);

namespace {

struct LargeSwappable {
  LargeSwappable() : d_(1024, 17) {}
  ~LargeSwappable() {}
  LargeSwappable(const LargeSwappable& o) : d_(o.d_) {}

  friend void swap(LargeSwappable& a, LargeSwappable& b) {
    using std::swap;
    swap(a.d_, b.d_);
  }

  LargeSwappable& operator=(LargeSwappable o) {
    using std::swap;
    swap(*this, o);
    return *this;
  }

  std::vector<int> d_;
};

}  // namespace

static void BM_LargeSwappableElements(int iters, int len) {
  typedef tensorflow::gtl::InlinedVector<LargeSwappable, 32> Vec;
  Vec a(len);
  Vec b;
  while (--iters >= 0) {
    using std::swap;
    swap(a, b);
  }
}
BENCHMARK(BM_LargeSwappableElements)->Range(0, 1024);

}  // namespace tensorflow
