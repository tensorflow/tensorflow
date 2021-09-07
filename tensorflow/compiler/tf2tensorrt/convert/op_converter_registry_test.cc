#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace tensorrt {
namespace convert {
TEST(TestOpConverterRegistry, TestOpConverterRegistry) {
  bool flag{false};

  auto set_true_func = [&flag](OpConverterParams*) -> Status {
    flag = true;
    return Status::OK();
  };

  auto set_false_func = [&flag](OpConverterParams*) -> Status {
    flag = false;
    return Status::OK();
  };

  GetOpConverterRegistry()->Register("FakeFunc", kDefaultConverterPriority,
                                     set_true_func);

  // Lower priority fails to override.
  GetOpConverterRegistry()->Register("FakeFunc", kDefaultConverterPriority - 1,
                                     set_false_func);

  // The lookup should return set_true_func (default).
  auto func = GetOpConverterRegistry()->LookUp("FakeFunc");
  ASSERT_TRUE(func.ok());
  (*func)(nullptr);
  ASSERT_TRUE(flag);

  // Override with higher priority.
  GetOpConverterRegistry()->Register("FakeFunc", kDefaultConverterPriority + 1,
                                     set_false_func);
  func = GetOpConverterRegistry()->LookUp("FakeFunc");
  ASSERT_TRUE(func.ok());
  (*func)(nullptr);
  ASSERT_FALSE(flag);

  // After clearing the op, lookup should return an error.
  GetOpConverterRegistry()->Clear("FakeFunc");
  ASSERT_FALSE(GetOpConverterRegistry()->LookUp("FakeFunc").ok());
}
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
