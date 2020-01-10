#include "tensorflow/core/util/saved_tensor_slice_util.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace checkpoint {

namespace {

// Testing serialization of tensor name and tensor slice in the ordered code
// format.
TEST(TensorShapeUtilTest, TensorNameSliceToOrderedCode) {
  {
    TensorSlice s = TensorSlice::ParseOrDie("-:-:1,3:4,5");
    string buffer = EncodeTensorNameSlice("foo", s);
    string name;
    s.Clear();
    TF_CHECK_OK(DecodeTensorNameSlice(buffer, &name, &s));
    EXPECT_EQ("foo", name);
    EXPECT_EQ("-:-:1,3:4,5", s.DebugString());
  }
}

}  // namespace

}  // namespace checkpoint

}  // namespace tensorflow
