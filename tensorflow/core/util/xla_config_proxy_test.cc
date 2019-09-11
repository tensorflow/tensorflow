/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/xla_config_proxy.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

enum ConfigA { A_ON, A_OFF };

enum ConfigB { B_ON, B_OFF };

};  // anonymous

TEST(XlaConfigProxyTest, RegistrationsAndQueries) {
  ConfigA config_a = A_ON;
  ConfigB config_b = B_ON;

  XlaConfigProxy::ConfigSetterRegistry<ConfigA>::Global()->Update(config_a);
  XlaConfigProxy::ConfigSetterRegistry<ConfigB>::Global()->Update(config_b);

  // Expect original values.
  EXPECT_EQ(A_ON, config_a);
  EXPECT_EQ(B_ON, config_b);

  // Register callbacks.
  XlaConfigProxy::ConfigSetterRegistration<ConfigA>
  config_setter_registration_a([](ConfigA& config_a) -> bool {
    config_a = A_OFF;
    return true;
  });

  XlaConfigProxy::ConfigSetterRegistration<ConfigB>
  config_setter_registration_b([](ConfigB& config_b) -> bool {
    config_b = B_OFF;
    return true;
  });

  // Verify that callbacks work.
  XlaConfigProxy::ConfigSetterRegistry<ConfigA>::Global()->Update(config_a);
  XlaConfigProxy::ConfigSetterRegistry<ConfigB>::Global()->Update(config_b);
  EXPECT_EQ(A_OFF, config_a);
  EXPECT_EQ(B_OFF, config_b);
}

}  // namespace tensorflow
