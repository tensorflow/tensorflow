/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/summary_optimizer.h"

#include <algorithm>
#include <string>
#include <vector>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::tensorflow::summary_optimizer::GetDisableSummariesInputArg;
using ::tensorflow::summary_optimizer::StrippedFunctionName;
using ::tensorflow::summary_optimizer::StripSummaries;
using ::tensorflow::summary_optimizer::internal::NormalizeEdgeName;
using ::tsl::protobuf::TextFormat;
using ::tsl::protobuf::util::MessageDifferencer;

template <typename T>
void CompareProto(const T& expected, const std::string& text_proto) {
  T proto;
  ASSERT_TRUE(TextFormat::ParseFromString(text_proto, &proto));
  MessageDifferencer differencer;
  EXPECT_TRUE(differencer.Compare(expected, proto));
}

TEST(SummaryOptimizerInternal, NormalizesEdgeName) {
  EXPECT_EQ(NormalizeEdgeName("include_summary"), "include_summary");
  EXPECT_EQ(NormalizeEdgeName("^include_summary"), "include_summary");
  EXPECT_EQ(NormalizeEdgeName("^include_summary:0"), "include_summary");
  EXPECT_EQ(NormalizeEdgeName("^include_summary/identity:0"),
            "include_summary/identity");
}

TEST(SummaryOptimizer, GetsDisableSummariesInputArg) {
  FunctionDef fdef;
  // When no disable_summaries_at_runtime attr is populated expect an empty str.
  auto input_arg = GetDisableSummariesInputArg(fdef);
  EXPECT_EQ(input_arg.first, "");
  EXPECT_FALSE(input_arg.second);

  AttrValue attr_val;
  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            list { s: "remove_summary" b: true }
                                          )pb",
                                          &attr_val));
  fdef.mutable_attr()->insert({"disable_summaries_at_runtime", attr_val});
  input_arg = GetDisableSummariesInputArg(fdef);
  EXPECT_EQ(input_arg.first, "remove_summary");
  EXPECT_TRUE(input_arg.second);
}

TEST(SummaryOptimizer, StripsSummaries) {
  FunctionDef fdef;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        signature {
          name: "train"  # Function name should be updated.
          input_arg: { name: "include_summaries" }
          control_output: "out_pruned"  # Control output should be pruned
                                        # because it was pruned from
                                        # `control_ret`.
          control_output: "out"
        }
        node_def { name: "x" }
        node_def {
          name: "write_summary/Identity"
        }  # Node should get pruned based on name.
        node_def {
          name: "Identity/x"
          input: "write_summary/Identity"  # Summary scope input should get
                                           # pruned.
          input: "x"
        }
        node_def {
          name: "nested_fn"
          op: "PartitionedCall"
          attr {
            key: "f"
            value: { func: { name: "nested_fn" } }
          }
        }
        node_def {
          name: "list_of_nested_fns"
          op: "SomeCustomOp"
          attr {
            key: "functions"
            value: {
              list: {
                func: { name: "nested_fn2" }
                func: { name: "nested_fn3" }
              }
            }
          }
        }
        node_def {
          op: "FlushSummaryWriter"
        }  # Node should get pruned based on op.
        control_ret {
          key: "out_pruned",
          value: "write_summary/Identity:0"
        }  # Control return should get pruned because node was pruned.
        control_ret { key: "out", value: "Identity/x" }
        attr {
          key: "forward_function_name"
          value: {
            s: "__inference_train_1"
          }  # Forward function name should be updated.
        }
        attr {
          key: "backward_function_name"
          value: {
            s: "__inference_train_2"
          }  # Backward function name should be updated.
        }
        attr {
          key: "disable_summaries_at_runtime"
          value: { list { s: "include_summaries" b: false } }
        }
      )pb",
      &fdef));
  FunctionDef nested_fdef;
  nested_fdef.mutable_signature()->set_name("nested_fn");
  FunctionDef nested_fdef2;
  nested_fdef2.mutable_signature()->set_name("nested_fn2");
  FunctionDef nested_fdef3;
  nested_fdef3.mutable_signature()->set_name("nested_fn3");

  FunctionLibraryDefinition flib(OpRegistry::Global());
  TF_ASSERT_OK(flib.AddFunctionDef(fdef));
  TF_ASSERT_OK(flib.AddFunctionDef(nested_fdef));
  TF_ASSERT_OK(flib.AddFunctionDef(nested_fdef2));
  TF_ASSERT_OK(flib.AddFunctionDef(nested_fdef3));

  std::vector<FunctionDef> stripped_fdefs = StripSummaries(fdef, flib);
  ASSERT_EQ(stripped_fdefs.size(), 4);
  // Sort the FunctionDefs so we are able to compare them in a deterministic
  // order.
  struct {
    bool operator()(const FunctionDef& lhs, const FunctionDef& rhs) const {
      return lhs.signature().name() > rhs.signature().name();
    }
  } fdefOrdering;
  std::sort(stripped_fdefs.begin(), stripped_fdefs.end(), fdefOrdering);
  CompareProto(stripped_fdefs[0], R"pb(
    signature {
      name: "train__instance__no_summaries"
      input_arg: { name: "include_summaries" }
      control_output: "out"
    }
    node_def { name: "x" }
    node_def { name: "Identity/x" input: "x" }
    node_def {
      name: "nested_fn"
      op: "PartitionedCall"
      attr {
        key: "f"
        value: { func: { name: "nested_fn__instance__no_summaries" } }
      }
    }
    node_def {
      name: "list_of_nested_fns"
      op: "SomeCustomOp"
      attr {
        key: "functions"
        value: {
          list: {
            func: { name: "nested_fn2__instance__no_summaries" }
            func: { name: "nested_fn3__instance__no_summaries" }
          }
        }
      }
    }
    control_ret { key: "out", value: "Identity/x" }
    attr {
      key: "forward_function_name",
      value: { s: "__inference_train_1__instance__no_summaries" }
    }
    attr {
      key: "backward_function_name",
      value: { s: "__inference_train_2__instance__no_summaries" }
    }
    attr {
      key: "disable_summaries_at_runtime"
      value {}
    }
  )pb");
  CompareProto(stripped_fdefs[1], R"pb(
    signature { name: "nested_fn__instance__no_summaries" }
  )pb");
  CompareProto(stripped_fdefs[2], R"pb(
    signature { name: "nested_fn3__instance__no_summaries" }
  )pb");
  CompareProto(stripped_fdefs[3], R"pb(
    signature { name: "nested_fn2__instance__no_summaries" }
  )pb");
}

TEST(SummaryOptimizer, DoesNotStripSummariesWhenNotEnabled) {
  FunctionDef fdef;
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(
                                    signature { name: "train" }
                                    attr {
                                      key: "disable_summaries_at_runtime",
                                      value: {}
                                    }
                                  )pb",
                                  &fdef));
  FunctionLibraryDefinition flib(OpRegistry::Global());
  TF_ASSERT_OK(flib.AddFunctionDef(fdef));

  // No stripped FunctionDefs generated when disable_summaries_at_runtime has no
  // value.
  EXPECT_TRUE(StripSummaries(fdef, flib).empty());

  // No stripped FunctionDefs generated when there is no
  // `disable_summaries_at_runtime` attr in the FunctionDef.
  fdef.clear_attr();
  TF_ASSERT_OK(flib.RemoveFunction("train"));
  TF_ASSERT_OK(flib.AddFunctionDef(fdef));
  EXPECT_TRUE(StripSummaries(fdef, flib).empty());
}

TEST(SummaryOptimizer, GeneratesNewFunctionName) {
  EXPECT_EQ(StrippedFunctionName("train"), "train__instance__no_summaries");
}

}  // namespace
}  // namespace tensorflow
