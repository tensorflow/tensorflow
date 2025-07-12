#include "tensorflow/dtensor/cc/small_constant_optimization.h"

#include "tensorflow/core/framework/node_def.pb.h"  // 只用 proto
#include "tensorflow/core/platform/test.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {
namespace {

TEST(SmallConstantOptimization, NullTensorReturnsNullopt) {
  // Arrange
  Layout layout = Layout::Empty();
  TF_Status* tf_status = TF_NewStatus();

  // Act
  std::optional<NodeDef> result = ExtractSmallTensorValue(
      /*context=*/nullptr,
      /*tensor=*/nullptr, layout, tf_status);

  // Assert
  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(TF_GetCode(tf_status), TF_INVALID_ARGUMENT);

  TF_DeleteStatus(tf_status);
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
