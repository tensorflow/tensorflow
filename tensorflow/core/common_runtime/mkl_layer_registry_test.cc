/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/common_runtime/mkl_layer_registry.h"

namespace tensorflow {

class MklLayerRegistrationTest : public ::testing::Test {
 public:
    MklLayerRegistrationTest() {}
    ~MklLayerRegistrationTest() {}
};

// Positive test with name and type registratation
TEST_F(MklLayerRegistrationTest, Positive_OpNameTypeRegistrationTest) {
  // Register Foo with FLOAT
  MklLayerRegistrar r("Foo", DT_FLOAT);

  // Check positive - Foo with FLOAT
  EXPECT_EQ(IS_MKL_LAYER("Foo", DT_FLOAT),  true);
  // Check Negative - Foo with DOUBLE
  EXPECT_EQ(IS_MKL_LAYER("Foo", DT_DOUBLE), false);

  // Clear registry for next test.
  MklLayerRegistry::Instance()->Clear();
}

// Same test as above with multiple ops and types.
TEST_F(MklLayerRegistrationTest, Positive_MultipleOpNameTypeRegistrationTest) {
  MklLayerRegistrar r1("Foo", DT_FLOAT);
  MklLayerRegistrar r2("Bar", DT_DOUBLE);

  EXPECT_EQ(IS_MKL_LAYER("Foo", DT_FLOAT),  true);
  EXPECT_EQ(IS_MKL_LAYER("Bar", DT_DOUBLE), true);

  EXPECT_EQ(IS_MKL_LAYER("Foo", DT_DOUBLE), false);
  EXPECT_EQ(IS_MKL_LAYER("Bar", DT_FLOAT),  false);
  EXPECT_EQ(IS_MKL_LAYER("Foo", DT_INT8),   false);
  EXPECT_EQ(IS_MKL_LAYER("Bar", DT_BOOL),   false);

  MklLayerRegistry::Instance()->Clear();
}

// Negative test for non existing (not registered) op
TEST_F(MklLayerRegistrationTest, Negative_OpNameTypeRegistrationTest) {
  MklLayerRegistrar r("Foo", DT_FLOAT);

  EXPECT_EQ(IS_MKL_LAYER("FooNonExist", DT_FLOAT),  false);
  EXPECT_EQ(IS_MKL_LAYER("FooNonExist", DT_DOUBLE), false);

  MklLayerRegistry::Instance()->Clear();
}

// Same tests as above but by using LayerRegistry API directly.

TEST_F(MklLayerRegistrationTest, Positive_OpNameTypeRegistrationAPITest) {
  MklLayerRegistry::Instance()->Register("Foo", DT_FLOAT);

  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Foo", DT_FLOAT),  true);
  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Foo", DT_DOUBLE), false);

  MklLayerRegistry::Instance()->Clear();
}

TEST_F(MklLayerRegistrationTest,
       Positive_MultipleOpNameTypeRegistrationAPITest) {
  MklLayerRegistry::Instance()->Register("Foo", DT_FLOAT);
  MklLayerRegistry::Instance()->Register("Bar", DT_DOUBLE);

  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Foo", DT_FLOAT),  true);
  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Bar", DT_DOUBLE), true);

  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Foo", DT_DOUBLE), false);
  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Bar", DT_FLOAT),  false);
  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Foo", DT_INT8),   false);
  EXPECT_EQ(MklLayerRegistry::Instance()->Find("Bar", DT_BOOL),   false);

  MklLayerRegistry::Instance()->Clear();
}

TEST_F(MklLayerRegistrationTest, Negative_OpNameTypeRegistrationAPITest) {
  MklLayerRegistry::Instance()->Register("Foo", DT_FLOAT);

  EXPECT_EQ(MklLayerRegistry::Instance()->Find("FooNonExist", DT_FLOAT),
            false);
  EXPECT_EQ(MklLayerRegistry::Instance()->Find("FooNonExist", DT_DOUBLE),
            false);

  MklLayerRegistry::Instance()->Clear();
}

}  // namespace tensorflow

#endif /* INTEL_MKL */
