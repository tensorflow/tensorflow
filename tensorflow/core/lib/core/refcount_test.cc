#include "tensorflow/core/lib/core/refcount.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace core {
namespace {

static int constructed = 0;
static int destroyed = 0;

class MyRef : public RefCounted {
 public:
  MyRef() { constructed++; }
  ~MyRef() override { destroyed++; }
};

class RefTest : public testing::Test {
 public:
  RefTest() {
    constructed = 0;
    destroyed = 0;
  }
};

TEST_F(RefTest, New) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(0, destroyed);
  ref->Unref();
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(1, destroyed);
}

TEST_F(RefTest, RefUnref) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(0, destroyed);
  ref->Ref();
  ASSERT_EQ(0, destroyed);
  ref->Unref();
  ASSERT_EQ(0, destroyed);
  ref->Unref();
  ASSERT_EQ(1, destroyed);
}

TEST_F(RefTest, RefCountOne) {
  MyRef* ref = new MyRef;
  ASSERT_TRUE(ref->RefCountIsOne());
  ref->Unref();
}

TEST_F(RefTest, RefCountNotOne) {
  MyRef* ref = new MyRef;
  ref->Ref();
  ASSERT_FALSE(ref->RefCountIsOne());
  ref->Unref();
  ref->Unref();
}

TEST_F(RefTest, ConstRefUnref) {
  const MyRef* cref = new MyRef;
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(0, destroyed);
  cref->Ref();
  ASSERT_EQ(0, destroyed);
  cref->Unref();
  ASSERT_EQ(0, destroyed);
  cref->Unref();
  ASSERT_EQ(1, destroyed);
}

TEST_F(RefTest, ReturnOfUnref) {
  MyRef* ref = new MyRef;
  ref->Ref();
  EXPECT_FALSE(ref->Unref());
  EXPECT_TRUE(ref->Unref());
}

TEST_F(RefTest, ScopedUnref) {
  { ScopedUnref unref(new MyRef); }
  EXPECT_EQ(destroyed, 1);
}

TEST_F(RefTest, ScopedUnref_Nullptr) {
  { ScopedUnref unref(nullptr); }
  EXPECT_EQ(destroyed, 0);
}

}  // namespace
}  // namespace core
}  // namespace tensorflow
