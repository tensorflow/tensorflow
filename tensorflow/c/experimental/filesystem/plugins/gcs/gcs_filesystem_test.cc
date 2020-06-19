#include "gtest/gtest.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status_helper.h"

#define ASSERT_TF_OK(x) ASSERT_EQ(TF_OK, TF_GetCode(x))

// Forward declaration
namespace tf_gcs_filesystem {
void Init(TF_Filesystem* filesystem, TF_Status* status);
}

namespace tensorflow {
namespace {

class GCSFilesystemTest : public ::testing::Test {
 public:
  void SetUp() override {
    status = TF_NewStatus();
    filesystem = new TF_Filesystem;
    tf_gcs_filesystem::Init(filesystem, status);
    ASSERT_TF_OK(status) << "Can not initialize filesystem. "
                         << TF_Message(status);
  }
  void TearDown() override {
    TF_DeleteStatus(status);
    // TODO(vnvo2409): Add filesystem cleanup
    delete filesystem;
  }

 protected:
  TF_Filesystem* filesystem;
  TF_Status* status;
};

// We have to add this test here because there must be at least one test.
// This test will be removed in the future.
TEST_F(GCSFilesystemTest, TestInit) { ASSERT_TF_OK(status); }

}  // namespace
}  // namespace tensorflow

GTEST_API_ int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
