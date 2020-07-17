/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/string_view.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace {

SavedObjectGraph ParseSavedObjectGraph(StringPiece text_proto) {
  SavedObjectGraph value;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(string(text_proto),
                                                          &value));
  return value;
}

constexpr absl::string_view kSingleChildFoo = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
)";

constexpr absl::string_view kSingleChildFooWithFuncBar = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 2
    local_name: "bar"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  function {
    concrete_functions: "__inference_my_func_5"
    function_spec {
      fullargspec {
        named_tuple_value {
          name: "FullArgSpec"
          values {
            key: "args"
            value {
              list_value {
              }
            }
          }
          values {
            key: "varargs"
            value {
              none_value {
              }
            }
          }
          values {
            key: "varkw"
            value {
              none_value {
              }
            }
          }
          values {
            key: "defaults"
            value {
              none_value {
              }
            }
          }
          values {
            key: "kwonlyargs"
            value {
              list_value {
              }
            }
          }
          values {
            key: "kwonlydefaults"
            value {
              none_value {
              }
            }
          }
          values {
            key: "annotations"
            value {
              dict_value {
              }
            }
          }
        }
      }
      input_signature {
        tuple_value {
        }
      }
    }
  }
}
concrete_functions {
  key: "__inference_my_func_5"
  value {
    canonicalized_input_signature {
      tuple_value {
        values {
          tuple_value {
          }
        }
        values {
          dict_value {
          }
        }
      }
    }
    output_signature {
      tensor_spec_value {
        shape {
        }
        dtype: DT_FLOAT
      }
    }
  }
}
)";

// In this graph, foo.baz and bar.wombat should point to the same object.
constexpr absl::string_view kMultiplePathsToChild = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  children {
    node_id: 2
    local_name: "bar"
  }
  children {
    node_id: 3
    local_name: "signatures"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 4
    local_name: "baz"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 4
    local_name: "wombat"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "signature_map"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
)";

// `foo` has edge `bar`, which has edge `parent` pointing back to `foo`.
constexpr absl::string_view kCycleBetweenParentAndChild = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  children {
    node_id: 2
    local_name: "signatures"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 3
    local_name: "bar"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "signature_map"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 1
    local_name: "parent"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
)";

TEST(ObjectGraphTraversalTest, Success) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kSingleChildFoo);
  const SavedObject* obj = internal::FindNodeAtPath("foo", object_graph);
  ASSERT_NE(nullptr, obj);
  EXPECT_EQ(obj->kind_case(), SavedObject::kUserObject);
  EXPECT_EQ(obj->user_object().identifier(), "_generic_user_object");
}

TEST(ObjectGraphTraversalTest, ObjectNotFound) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kSingleChildFoo);
  const SavedObject* obj = internal::FindNodeAtPath("bar", object_graph);
  EXPECT_EQ(nullptr, obj);
}

TEST(ObjectGraphTraversalTest, CaseSensitiveMismatch) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kSingleChildFoo);
  const SavedObject* obj = internal::FindNodeAtPath("FOO", object_graph);
  EXPECT_EQ(nullptr, obj);
}

TEST(ObjectGraphTraversalTest, NestedObjectFound) {
  SavedObjectGraph object_graph =
      ParseSavedObjectGraph(kSingleChildFooWithFuncBar);
  const SavedObject* obj = internal::FindNodeAtPath("foo.bar", object_graph);
  ASSERT_NE(nullptr, obj);
  EXPECT_EQ(obj->kind_case(), SavedObject::kFunction);
  EXPECT_EQ(obj->function().concrete_functions_size(), 1);
  EXPECT_EQ(obj->function().concrete_functions(0), "__inference_my_func_5");
}

TEST(ObjectGraphTraversalTest, MultiplePathsAliasSameObject) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kMultiplePathsToChild);
  const SavedObject* foo_baz =
      internal::FindNodeAtPath("foo.baz", object_graph);
  ASSERT_NE(nullptr, foo_baz);
  EXPECT_EQ(foo_baz->kind_case(), SavedObject::kUserObject);
  EXPECT_EQ(foo_baz->user_object().identifier(), "_generic_user_object");

  const SavedObject* bar_wombat =
      internal::FindNodeAtPath("bar.wombat", object_graph);
  ASSERT_NE(nullptr, bar_wombat);
  EXPECT_EQ(bar_wombat->kind_case(), SavedObject::kUserObject);
  EXPECT_EQ(bar_wombat->user_object().identifier(), "_generic_user_object");

  EXPECT_EQ(foo_baz, bar_wombat);
}

TEST(ObjectGraphTraversalTest, CyclesAreOK) {
  SavedObjectGraph object_graph =
      ParseSavedObjectGraph(kCycleBetweenParentAndChild);
  const SavedObject* foo = internal::FindNodeAtPath("foo", object_graph);
  ASSERT_NE(nullptr, foo);
  EXPECT_EQ(foo->kind_case(), SavedObject::kUserObject);
  EXPECT_EQ(foo->user_object().identifier(), "_generic_user_object");

  const SavedObject* foo_bar =
      internal::FindNodeAtPath("foo.bar", object_graph);
  ASSERT_NE(nullptr, foo_bar);
  EXPECT_EQ(foo_bar->kind_case(), SavedObject::kUserObject);
  EXPECT_EQ(foo_bar->user_object().identifier(), "_generic_user_object");

  const SavedObject* foo_bar_parent =
      internal::FindNodeAtPath("foo.bar.parent", object_graph);
  ASSERT_NE(nullptr, foo_bar_parent);
  EXPECT_EQ(foo_bar_parent->kind_case(), SavedObject::kUserObject);
  EXPECT_EQ(foo_bar_parent->user_object().identifier(), "_generic_user_object");

  const SavedObject* foo_bar_parent_bar =
      internal::FindNodeAtPath("foo.bar.parent.bar", object_graph);
  ASSERT_NE(nullptr, foo_bar_parent_bar);
  EXPECT_EQ(foo_bar_parent_bar->kind_case(), SavedObject::kUserObject);
  EXPECT_EQ(foo_bar_parent_bar->user_object().identifier(),
            "_generic_user_object");

  EXPECT_EQ(foo, foo_bar_parent);
  EXPECT_EQ(foo_bar, foo_bar_parent_bar);
}

}  // namespace
}  // namespace tensorflow
