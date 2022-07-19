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

#include "tensorflow/core/framework/node_properties.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

OpDef ToOpDef(const OpDefBuilder& builder) {
  OpRegistrationData op_reg_data;
  EXPECT_TRUE(builder.Finalize(&op_reg_data).ok());
  return op_reg_data.op_def;
}

class MockOpRegistry : public OpRegistryInterface {
 public:
  MockOpRegistry()
      : op_reg_(ToOpDef(OpDefBuilder("Foo")
                            .Input("f: float")
                            .Input("i: int32")
                            .Output("of: double"))) {}
  ~MockOpRegistry() override {}

  // Returns an error status and sets *op_reg_data to nullptr if no OpDef is
  // registered under that name, otherwise returns the registered OpDef.
  // Caller must not delete the returned pointer.
  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override {
    if (op_type_name == "Foo") {
      *op_reg_data = &op_reg_;
      return OkStatus();
    } else {
      *op_reg_data = nullptr;
      return errors::InvalidArgument("Op type named ", op_type_name,
                                     " not found");
    }
  }

  const OpDef* get_op_def_addr() { return &op_reg_.op_def; }

 private:
  const OpRegistrationData op_reg_;
};

void ValidateNodeProperties(const NodeProperties& props, const OpDef* op_def,
                            const NodeDef& node_def,
                            const DataTypeVector& input_types,
                            const DataTypeVector& output_types) {
  EXPECT_EQ(props.op_def, op_def);
  EXPECT_EQ(props.node_def.name(), node_def.name());
  ASSERT_EQ(props.input_types.size(), input_types.size());
  for (int i = 0; i < input_types.size(); ++i) {
    EXPECT_EQ(props.input_types[i], input_types[i]);
    EXPECT_EQ(props.input_types_slice[i], input_types[i]);
  }
  ASSERT_EQ(props.output_types.size(), output_types.size());
  for (int i = 0; i < output_types.size(); ++i) {
    EXPECT_EQ(props.output_types[i], output_types[i]);
    EXPECT_EQ(props.output_types_slice[i], output_types[i]);
  }
}

}  // namespace

TEST(NodeProperties, Contructors) {
  OpDef op_def;
  NodeDef node_def;
  node_def.set_name("foo");
  DataTypeVector input_types{DT_FLOAT, DT_INT32};
  DataTypeVector output_types{DT_DOUBLE};
  DataTypeSlice input_types_slice(input_types);
  DataTypeSlice output_types_slice(output_types);

  // Construct from slices.
  NodeProperties props_from_slices(&op_def, node_def, input_types_slice,
                                   output_types_slice);
  ValidateNodeProperties(props_from_slices, &op_def, node_def, input_types,
                         output_types);

  // Construct from vectors.
  NodeProperties props_from_vectors(&op_def, node_def, input_types,
                                    output_types);
  ValidateNodeProperties(props_from_vectors, &op_def, node_def, input_types,
                         output_types);
}

TEST(NodeProperties, CreateFromNodeDef) {
  MockOpRegistry op_registry;
  NodeDef node_def;
  node_def.set_name("bar");
  node_def.set_op("Foo");
  node_def.add_input("f_in");
  node_def.add_input("i_in");

  std::shared_ptr<const NodeProperties> props;
  EXPECT_TRUE(
      NodeProperties::CreateFromNodeDef(node_def, &op_registry, &props).ok());

  DataTypeVector input_types{DT_FLOAT, DT_INT32};
  DataTypeVector output_types{DT_DOUBLE};
  ValidateNodeProperties(*props, op_registry.get_op_def_addr(), node_def,
                         input_types, output_types);

  // The OpDef lookup should fail for this one:
  node_def.set_op("Baz");
  std::shared_ptr<const NodeProperties> props_bad;
  EXPECT_FALSE(
      NodeProperties::CreateFromNodeDef(node_def, &op_registry, &props_bad)
          .ok());
  EXPECT_EQ(props_bad, nullptr);
}
}  // namespace tensorflow
