/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/kernel_def_util.h"

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

NodeDef NodeDefFromText(const string& text) {
  NodeDef node_def;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &node_def));
  return node_def;
}

KernelDef KernelDefFromText(const string& text) {
  KernelDef kernel_def;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &kernel_def));
  return kernel_def;
}

class AttrsMatchTest : public ::testing::Test {
 protected:
  void ExpectStatus(const string& node_def_str, const string& kernel_def_str,
                    error::Code code) {
    bool match;
    auto status = KernelAttrsMatch(KernelDefFromText(kernel_def_str),
                                   NodeDefFromText(node_def_str), &match);
    LOG(INFO) << "status: " << status;
    EXPECT_EQ(code, status.code());
    if (!status.ok()) {
      EXPECT_FALSE(match)
          << "Expect no match between the given NodeDef and KernelDef";
    }
  }
};

TEST_F(AttrsMatchTest, ValidConstraint) {
  string node_def_str = R"(
    name: "ValidConstraint-op"
    op: "ValidConstraint"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
  )";
  string kernel_def_str = R"(
    op: "ValidConstraint"
    device_type: "CPU"
    constraint {
      name: "T"
      allowed_values {
        list {
          type: DT_FLOAT
        }
      }
    }
  )";
  ExpectStatus(node_def_str, kernel_def_str, error::OK);
}

TEST_F(AttrsMatchTest, BadConstraint) {
  string node_def_str = R"(
    name: "BadConstraint-op"
    op: "BadConstraint"
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
  )";
  string kernel_def_str = R"(
    op: "BadConstraint"
    device_type: "CPU"
    constraint {
      name: "T"
      allowed_values {
        list {
          type: DT_FLOAT
        }
      }
    }
  )";
  ExpectStatus(node_def_str, kernel_def_str, error::INVALID_ARGUMENT);
}

TEST_F(AttrsMatchTest, Unimplemented) {
  string node_def_str = R"(
    name: "BadConstraint-op"
    op: "BadConstraint"
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
  )";
  string kernel_def_str = R"(
    op: "BadConstraint"
    device_type: "CPU"
    constraint {
      name: "T"
      allowed_values {
        list {
        }
      }
    }
  )";
  ExpectStatus(node_def_str, kernel_def_str, error::UNIMPLEMENTED);
}

}  // namespace
}  // namespace tensorflow
