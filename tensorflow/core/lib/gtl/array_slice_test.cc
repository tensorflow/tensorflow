#include "tensorflow/core/lib/gtl/array_slice.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/port.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace gtl {
namespace {

typedef ArraySlice<int> IntSlice;
typedef ArraySlice<char> CharSlice;
typedef MutableArraySlice<int> MutableIntSlice;
typedef MutableArraySlice<char> MutableCharSlice;
typedef std::vector<int> IntVec;

// Append 0..len-1 to *v
template <typename Vector>
static void Fill(Vector* v, int len, int offset = 0) {
  for (int i = 0; i < len; i++) {
    v->push_back(i + offset);
  }
}

static void TestHelper(const IntSlice& vorig, const IntVec& vec) {
  IntSlice other;  // To test the assignment return value.
  IntSlice v = other = vorig;
  const int len = vec.size();
  EXPECT_EQ(v.size(), vec.size());

  for (int i = 0; i < len; i++) {
    EXPECT_EQ(v[i], vec[i]);
    EXPECT_EQ(v.at(i), vec[i]);
  }
  EXPECT_EQ(v.begin(), gtl::vector_as_array(&vec));

  int counter = 0;
  for (IntSlice::iterator it = v.begin(); it != v.end(); ++it) {
    EXPECT_EQ(counter, *it);
    counter++;
  }
  EXPECT_EQ(counter, len);

  counter = 0;
  for (IntSlice::const_iterator it = v.begin(); it != v.end(); ++it) {
    EXPECT_EQ(counter, *it);
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
    if (len > 1) {
      v.pop_front();
      EXPECT_EQ(len - 2, v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(i + 1, v[i]);
      }
    }
  }
}

// The element access test that is applicable both when MutableArraySlice is
// const and when it's not.
template <class V>
void MutableTestHelperTemplated(V v, int* ptr, const int len) {
  CHECK_EQ(v.size(), len);

  for (int i = 0; i < len; i++) {
    EXPECT_EQ(ptr + i, &v[i]);
    EXPECT_EQ(ptr + i, &v.at(i));
  }
  EXPECT_EQ(ptr, v.begin());
  EXPECT_EQ(ptr + len, v.end());
  EXPECT_EQ(ptr, v.data());

  int counter = 0;
  for (MutableIntSlice::const_iterator it = v.begin(); it != v.end(); ++it) {
    EXPECT_EQ(ptr + counter, &*it);
    counter++;
  }
  EXPECT_EQ(counter, len);

  EXPECT_EQ(len, std::distance(v.rbegin(), v.rend()));

  if (len > 0) {
    EXPECT_EQ(ptr, &v.front());
    EXPECT_EQ(ptr + len - 1, &v.back());
    EXPECT_EQ(ptr + len - 1, &*v.rbegin());
    EXPECT_EQ(ptr, &*(v.rend() - 1));
  }
}

static void MutableTestHelper(const MutableIntSlice& vorig, int* ptr,
                              const int len) {
  // Test the data accessors both when the MutableArraySlice is declared const,
  // and when it is not.
  MutableTestHelperTemplated<const MutableIntSlice&>(vorig, ptr, len);
  MutableTestHelperTemplated<MutableIntSlice>(vorig, ptr, len);

  MutableIntSlice other;  // To test the assignment return value.
  MutableIntSlice v = other = vorig;
  EXPECT_EQ(ptr, v.mutable_data());

  int counter = 0;
  for (MutableIntSlice::iterator it = v.begin(); it != v.end(); ++it) {
    EXPECT_EQ(ptr + counter, &*it);
    counter++;
  }
  EXPECT_EQ(counter, len);

  if (len > 0) {
    // Test that elements are assignable.
    v[0] = 1;
    v.front() = 2;
    v.back() = 5;
    *v.mutable_data() = 4;
    std::fill(v.begin(), v.end(), 5);
    std::fill(v.rbegin(), v.rend(), 6);
    // Test size-changing methods.
    v.pop_back();
    EXPECT_EQ(len - 1, v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      EXPECT_EQ(ptr + i, &v[i]);
    }
    if (len > 1) {
      v.pop_front();
      EXPECT_EQ(len - 2, v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(ptr + i + 1, &v[i]);
      }
    }
  }
}

template <typename Vector>
static void TestImplicitConversion(const IntSlice& v, const Vector& vec) {
  EXPECT_EQ(v.size(), vec.size());
  for (size_t i = 0; i < v.size(); i++) {
    EXPECT_EQ(v[i], vec[i]);
  }
}

template <typename Vector>
static void TestImplicitConversion(const CharSlice& v, const Vector& vec) {
  TestImplicitConversion(IntVec(v.begin(), v.end()), vec);
}

static void TestImplicitConversion(const MutableIntSlice& v, const int* data,
                                   int size) {
  EXPECT_EQ(size, v.size());
  for (size_t i = 0; i < v.size(); i++) {
    EXPECT_EQ(data + i, &v[i]);
  }
}

static void TestImplicitConversion(const MutableCharSlice& v, const char* data,
                                   int size) {
  EXPECT_EQ(size, v.size());
  for (size_t i = 0; i < v.size(); i++) {
    EXPECT_EQ(data + i, &v[i]);
  }
}
// A struct supplying the data(), mutable_data() and size() methods, just like
// e.g. proto2::RepeatedField.
struct RepeatedField {
  std::vector<int> storage;
  const int* data() const { return storage.data(); }
  int* mutable_data() { return storage.data(); }
  int size() const { return storage.size(); }
};

// A struct supplying the data() (both mutable and const versions) and
// size(). It also supplies mutable_data() but we test that data() is selected
// instead.
struct ContainerWithOverloads {
  std::vector<int> storage;
  std::vector<int> wrong_storage;
  const int* data() const { return storage.data(); }
  int* data() { return storage.data(); }
  // MutableArraySlice should not call mutable_data(), preferring data()
  // instead.
  int* mutable_data() { return wrong_storage.data(); }
  int size() const { return storage.size(); }
};

// A struct supplying data() and size() methods.
struct ContainerWithShallowConstData {
  std::vector<int> storage;
  int* data() const { return const_cast<int*>(storage.data()); }
  int size() const { return storage.size(); }
};

TEST(IntSlice, Simple) {
  for (int len = 0; len < 20; len++) {
    IntVec vec;
    Fill(&vec, len);
    TestHelper(IntSlice(vec), vec);
    TestHelper(IntSlice(vec.data(), vec.size()), vec);
  }
}

TEST(IntSlice, WithPosAndLen) {
  IntVec vec;
  Fill(&vec, 20);
  for (size_t len = 0; len < vec.size(); len++) {
    IntVec subvec(vec.begin(), vec.begin() + len);
    TestImplicitConversion(IntSlice(vec, 0, len), subvec);
    TestImplicitConversion(IntSlice(IntSlice(vec), 0, len), subvec);
  }
  EXPECT_EQ(0, IntSlice(vec, 0, 0).size());
  EXPECT_EQ(0, IntSlice(IntSlice(vec), 0, 0).size());
  TestImplicitConversion(IntSlice(vec, 0, IntSlice::npos), vec);
}

TEST(IntSlice, Clear) {
  for (int len = 0; len < 20; len++) {
    IntVec vec;
    Fill(&vec, len);
    IntSlice v(vec);
    v.clear();
    EXPECT_EQ(0, v.size());
    EXPECT_EQ(v.begin(), v.end());
  }
}

TEST(IntSlice, Swap) {
  for (int l1 = 0; l1 < 20; l1++) {
    for (int l2 = 0; l2 < 20; l2++) {
      IntVec avec, bvec;
      Fill(&avec, l1);
      Fill(&bvec, l2, 100);
      IntSlice a(avec), b(bvec);
      using std::swap;
      swap(a, b);
      EXPECT_EQ(l1, b.size());
      EXPECT_EQ(l2, a.size());
      for (int i = 0; i < l1; i++) {
        EXPECT_EQ(i, b[i]);
      }
      for (int i = 0; i < l2; i++) {
        EXPECT_EQ(100 + i, a[i]);
      }
    }
  }
}

TEST(IntSlice, ImplicitConversion) {
  for (int len = 0; len < 20; len++) {
    IntVec vec;
    Fill(&vec, len);
    IntSlice slice;
    slice = vec;
    TestImplicitConversion(vec, vec);
    TestImplicitConversion(slice, vec);
    TestImplicitConversion(IntSlice(vec.data(), vec.size()), vec);
  }
}

TEST(IntSlice, InlinedVectorConversion) {
  for (int len = 0; len < 20; len++) {
    InlinedVector<int, 4> inline_vec;
    for (int i = 0; i < len; i++) {
      inline_vec.push_back(i);
    }
    IntVec vec;
    Fill(&vec, len);
    IntSlice v = inline_vec;  // Test assignment
    static_cast<void>(v);
    TestImplicitConversion(inline_vec, vec);
  }
}

TEST(IntSlice, StaticArrayConversion) {
  int array[20];
  IntVec vec;
  Fill(&vec, TF_ARRAYSIZE(array));
  std::copy(vec.begin(), vec.end(), array);
  IntSlice v = array;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(array, vec);
}

TEST(IntSlice, StdArrayConversion) {
  std::array<int, 20> array;
  IntVec vec;
  Fill(&vec, array.size());
  std::copy(vec.begin(), vec.end(), array.begin());

  // Check assignment.
  {
    IntSlice v = array;
    static_cast<void>(v);
  }

  // Check sub-slice initialization.
  {
    IntSlice v = {array, 10, 15};
    static_cast<void>(v);
  }

  TestImplicitConversion(array, vec);
}

// Values according to the Fill function.
static const int test_const_array[] = {0, 1, 2};

TEST(IntSlice, ConstStaticArrayConversion) {
  IntVec vec;
  Fill(&vec, TF_ARRAYSIZE(test_const_array));
  IntSlice v = test_const_array;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(test_const_array, vec);
}

TEST(IntSlice, RepeatedFieldConversion) {
  RepeatedField repeated_field;
  IntVec vec;
  Fill(&vec, 20);
  repeated_field.storage = vec;
  IntSlice v = repeated_field;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(repeated_field, vec);
}

TEST(IntSlice, ContainerWithOverloadsConversion) {
  ContainerWithOverloads container;
  Fill(&container.storage, 20);
  container.wrong_storage.resize(container.size());
  IntSlice v = container;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(container, container.storage);
}

TEST(IntSlice, ContainerWithShallowConstDataConversion) {
  ContainerWithShallowConstData container;
  Fill(&container.storage, 20);
  IntSlice v = container;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(container, container.storage);
}

TEST(IntSlice, MutableIntSliceConversion) {
  IntVec vec(20);
  IntSlice slice = MutableIntSlice(&vec);
  EXPECT_EQ(vec.size(), slice.size());
  EXPECT_EQ(vec.data(), slice.data());
}

TEST(IntSlice, Equality) {
  IntVec vec1(20);
  IntVec vec2(20);
  // These two slices are from different vectors, but have the same
  // size and have the same elements (right now).  They should
  // compare equal.
  const IntSlice from1(vec1);
  const IntSlice from2(vec2);
  EXPECT_EQ(from1, from1);
  EXPECT_EQ(from1, from2);

  // This verifies that MutableArraySlices can be compared freely with
  // ArraySlices.
  const MutableIntSlice mutable_from1(&vec1);
  const MutableIntSlice mutable_from2(&vec2);
  EXPECT_EQ(from1, mutable_from1);
  EXPECT_EQ(mutable_from1, from1);
  EXPECT_EQ(mutable_from1, mutable_from2);
  EXPECT_EQ(mutable_from2, mutable_from1);

  // With a different size, the array slices should not be equal.
  EXPECT_NE(from1, IntSlice(from1, 0, from1.size() - 1));

  // With different contents, the array slices should not be equal.
  ++vec2.back();
  EXPECT_NE(from1, from2);
}

// Compile-asserts that the argument has the expected type.
template <typename Expected, typename T>
void CheckType(const T& value) {
  testing::StaticAssertTypeEq<Expected, T>();
}

TEST(IntSlice, ExposesContainerTypesAndConsts) {
  IntSlice slice;
  const IntSlice const_slice;
  CheckType<IntSlice::iterator>(slice.begin());
  CheckType<IntSlice::const_iterator>(const_slice.end());
  CheckType<IntSlice::const_reverse_iterator>(const_slice.rbegin());
  CheckType<IntSlice::reverse_iterator>(slice.rend());
  testing::StaticAssertTypeEq<int, IntSlice::value_type>();
  testing::StaticAssertTypeEq<const int*, IntSlice::pointer>();
  testing::StaticAssertTypeEq<const int&, IntSlice::const_reference>();
  EXPECT_EQ(static_cast<IntSlice::size_type>(-1), IntSlice::npos);
}

void TestEmpty(IntSlice slice) { ASSERT_TRUE(slice.empty()); }

void TestRange(IntSlice slice, int from, int to) {
  ASSERT_EQ(to - from + 1, slice.size());
  for (size_t i = 0; i < slice.size(); ++i) {
    EXPECT_EQ(from + i, slice[i]);
  }
}

TEST(IntSlice, InitializerListConversion) {
  TestEmpty({});
  TestRange({1}, 1, 1);
  TestRange({10, 11, 12, 13}, 10, 13);
}

TEST(CharSlice, StringConversion) {
  IntVec vec;
  Fill(&vec, 20);
  string str(vec.begin(), vec.end());
  CharSlice v = str;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(str, vec);
}

TEST(IntPtrSlice, ConstConversion) {
  int one = 1;
  int two = 2;
  std::vector<int*> vec;
  vec.push_back(&one);
  vec.push_back(&two);
  ArraySlice<const int*> v = vec;
  ASSERT_EQ(2, v.size());
  EXPECT_EQ(&one, v[0]);
  EXPECT_EQ(&two, v[1]);
}

TEST(MutableIntSlice, Simple) {
  for (int len = 0; len < 20; len++) {
    IntVec vec(len);
    MutableTestHelper(MutableIntSlice(&vec), vec.data(), len);
    MutableTestHelper(MutableIntSlice(vec.data(), vec.size()), vec.data(), len);
  }
}

TEST(MutableIntSlice, WithPosAndLen) {
  IntVec vec(20);
  for (size_t len = 0; len < vec.size(); len++) {
    TestImplicitConversion(MutableIntSlice(&vec, 0, len), vec.data(), len);
    TestImplicitConversion(MutableIntSlice(MutableIntSlice(&vec), 0, len),
                           vec.data(), len);
  }
  EXPECT_EQ(0, MutableIntSlice(&vec, 0, 0).size());
  EXPECT_EQ(0, MutableIntSlice(MutableIntSlice(&vec), 0, 0).size());
  TestImplicitConversion(MutableIntSlice(&vec, 0, MutableIntSlice::npos),
                         vec.data(), vec.size());
}

TEST(MutableIntSlice, Clear) {
  for (int len = 0; len < 20; len++) {
    IntVec vec(len);
    MutableIntSlice v(&vec);
    v.clear();
    EXPECT_EQ(0, v.size());
    EXPECT_EQ(v.begin(), v.end());
  }
}

TEST(MutableIntSlice, Swap) {
  for (int l1 = 0; l1 < 20; l1++) {
    for (int l2 = 0; l2 < 20; l2++) {
      IntVec avec(l1), bvec(l2);
      MutableIntSlice a(&avec), b(&bvec);
      using std::swap;
      swap(a, b);
      EXPECT_EQ(l1, b.size());
      EXPECT_EQ(l2, a.size());
      for (int i = 0; i < l1; i++) {
        EXPECT_EQ(&avec[i], &b[i]);
      }
      for (int i = 0; i < l2; i++) {
        EXPECT_EQ(&bvec[i], &a[i]);
      }
    }
  }
}

TEST(MutableIntSlice, ImplicitConversion) {
  for (int len = 0; len < 20; len++) {
    IntVec vec(len);
    MutableIntSlice slice;
    slice = &vec;
    TestImplicitConversion(&vec, vec.data(), len);
    TestImplicitConversion(slice, vec.data(), len);
    TestImplicitConversion(MutableIntSlice(vec.data(), vec.size()), vec.data(),
                           len);
  }
}

TEST(MutableIntSlice, InlinedVectorConversion) {
  for (int len = 0; len < 20; len++) {
    InlinedVector<int, 4> inline_vec;
    for (int i = 0; i < len; i++) {
      inline_vec.push_back(i);
    }
    MutableIntSlice v = &inline_vec;  // Test assignment
    static_cast<void>(v);
    TestImplicitConversion(&inline_vec, inline_vec.array(), inline_vec.size());
  }
}

TEST(MutableIntSlice, StaticArrayConversion) {
  int array[20];
  MutableIntSlice v = array;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(array, array, TF_ARRAYSIZE(array));
}

TEST(MutableIntSlice, StdArrayConversion) {
  std::array<int, 20> array;

  // Check assignment.
  {
    MutableIntSlice v = &array;
    static_cast<void>(v);
  }

  // Check sub-slice initialization.
  {
    MutableIntSlice v = {&array, 10, 15};
    static_cast<void>(v);
  }

  TestImplicitConversion(&array, &array[0], array.size());
}

TEST(MutableIntSlice, RepeatedFieldConversion) {
  RepeatedField repeated_field;
  Fill(&repeated_field.storage, 20);
  MutableIntSlice v = &repeated_field;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(&repeated_field, repeated_field.storage.data(),
                         repeated_field.storage.size());
}

TEST(MutableIntSlice, ContainerWithOverloadsConversion) {
  ContainerWithOverloads container;
  Fill(&container.storage, 20);
  container.wrong_storage.resize(container.size());
  MutableIntSlice v = &container;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(&container, container.storage.data(),
                         container.storage.size());
}

TEST(MutableIntSlice, ContainerWithShallowConstDataConversion) {
  ContainerWithShallowConstData container;
  Fill(&container.storage, 20);
  MutableIntSlice v = &container;  // Test assignment
  static_cast<void>(v);
  TestImplicitConversion(&container, container.storage.data(),
                         container.storage.size());
}

TEST(MutableIntSlice, TypedefsAndConstants) {
  testing::StaticAssertTypeEq<int, MutableIntSlice::value_type>();
  testing::StaticAssertTypeEq<int*, MutableIntSlice::pointer>();
  testing::StaticAssertTypeEq<const int*, MutableIntSlice::const_pointer>();
  testing::StaticAssertTypeEq<int&, MutableIntSlice::reference>();
  testing::StaticAssertTypeEq<const int&, MutableIntSlice::const_reference>();

  EXPECT_EQ(static_cast<MutableIntSlice::size_type>(-1), MutableIntSlice::npos);
}

TEST(MutableIntSlice, IteratorsAndReferences) {
  auto accept_pointer = [](int* x) {};
  auto accept_reference = [](int& x) {};
  auto accept_iterator = [](MutableIntSlice::iterator x) {};
  auto accept_reverse_iterator = [](MutableIntSlice::reverse_iterator x) {};

  int a[1];
  MutableIntSlice s = a;

  accept_pointer(s.data());
  accept_pointer(s.mutable_data());
  accept_iterator(s.begin());
  accept_iterator(s.end());
  accept_reverse_iterator(s.rbegin());
  accept_reverse_iterator(s.rend());

  accept_reference(s[0]);
  accept_reference(s.at(0));
  accept_reference(s.front());
  accept_reference(s.back());
}

TEST(MutableIntSlice, IteratorsAndReferences_Const) {
  auto accept_pointer = [](int* x) {};
  auto accept_reference = [](int& x) {};
  auto accept_iterator = [](MutableIntSlice::iterator x) {};
  auto accept_reverse_iterator = [](MutableIntSlice::reverse_iterator x) {};

  int a[1];
  const MutableIntSlice s = a;

  accept_pointer(s.data());
  accept_pointer(s.mutable_data());
  accept_iterator(s.begin());
  accept_iterator(s.end());
  accept_reverse_iterator(s.rbegin());
  accept_reverse_iterator(s.rend());

  accept_reference(s[0]);
  accept_reference(s.at(0));
  accept_reference(s.front());
  accept_reference(s.back());
}

bool TestMutableOverload(MutableIntSlice slice) { return false; }

bool TestMutableOverload(MutableCharSlice slice) { return true; }

TEST(MutableCharSlice, StringConversion) {
  for (int len = 0; len < 20; len++) {
    string str(len, '\0');
    MutableCharSlice v = &str;  // Test assignment
    static_cast<void>(v);
    TestImplicitConversion(v, str.data(), str.size());
  }
  // Verify that only the correct overload is feasible. Note that this would
  // fail if the string ctor was declared simply as MutableArraySlice(string*),
  // since in that case both overloads would be feasible.
  string str;
  EXPECT_TRUE(TestMutableOverload(&str));
}

}  // namespace
}  // namespace gtl
}  // namespace tensorflow
