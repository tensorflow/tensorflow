/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/translate/mlrt/mlir_to_bytecode.h"

#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/attribute_span.h"
#include "tsl/platform/resource_loader.h"
#include "tsl/platform/status_matchers.h"

namespace mlrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::IsEmpty;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(MlirToByteCodeTest, Basic) {
  constexpr char kBasicMlir[] =
      "tensorflow/compiler/mlir/tfrt/translate/mlrt/testdata/basic.mlir";

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext mlir_context(registry);
  mlir_context.allowUnregisteredDialects();
  auto mlir_module = mlir::parseSourceFile<mlir::ModuleOp>(
      tsl::GetDataDependencyFilepath(kBasicMlir), &mlir_context);

  AttributeEncoderRegistry attribute_encoder_registry;
  bc::Buffer buffer =
      EmitExecutable(attribute_encoder_registry, mlir_module.get()).value();

  bc::Executable executable(buffer.data());

  auto kernel_names = executable.kernel_names();
  EXPECT_THAT(kernel_names,
              ElementsAreArray({"test_mlbc.add.i32", "test_mlbc.sub.i32",
                                "call", "return"}));

  auto functions = executable.functions();
  ASSERT_GE(functions.size(), 1);

  auto function = functions[0];
  EXPECT_EQ(function.name().str(), "add_i32_10");
  EXPECT_EQ(function.num_regs(), 5);
  EXPECT_THAT(function.input_regs(), ElementsAreArray({0}));
  EXPECT_THAT(function.output_regs(), ElementsAreArray({0, 2, 2}));
  EXPECT_THAT(function.output_last_uses(),
              ElementsAreArray({true, false, true}));

  auto kernels = function.kernels();
  ASSERT_EQ(kernels.size(), 11);

  EXPECT_EQ(kernels[0].code(), 0);
  EXPECT_THAT(kernels[0].arguments(), ElementsAreArray({0, 0}));
  EXPECT_THAT(kernels[0].results(), ElementsAreArray({1}));
  EXPECT_THAT(kernels[0].last_uses(), ElementsAreArray({0, 0}));

  for (int i = 1; i < 9; i++) {
    EXPECT_EQ(kernels[i].code(), i % 2);
    EXPECT_THAT(kernels[i].arguments(), ElementsAreArray({(i - 1) % 2 + 1, 0}));
    EXPECT_THAT(kernels[i].results(), ElementsAreArray({i % 2 + 1}));
    EXPECT_THAT(kernels[i].last_uses(), ElementsAreArray({1, 0}));
  }

  EXPECT_EQ(kernels[9].code(), 2);
  EXPECT_THAT(kernels[9].arguments(), ElementsAreArray({1}));
  EXPECT_THAT(kernels[9].last_uses(), ElementsAreArray({true}));
  EXPECT_THAT(kernels[9].results(), ElementsAreArray({2, 3, 4}));

  EXPECT_EQ(kernels[10].code(), 3);
  EXPECT_THAT(kernels[10].arguments(), ElementsAreArray({0, 2, 2}));
  EXPECT_THAT(kernels[10].last_uses(), ElementsAreArray({true, false, true}));
  EXPECT_TRUE(kernels[10].results().empty());
}

template <typename T>
absl::StatusOr<T> DecodeAttribute(absl::string_view data) {
  if (data.size() < sizeof(T))
    return absl::InvalidArgumentError("Invalid data size for attribute.");

  T value;
  std::memcpy(&value, data.data(), sizeof(T));
  return value;
}

TEST(MlirToByteCodeTest, BasicAttributes) {
  constexpr char kBasicAttributesMlir[] =
      "tensorflow/compiler/mlir/tfrt/translate/mlrt/testdata/"
      "basic_attributes.mlir";

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext mlir_context(registry);
  mlir_context.allowUnregisteredDialects();
  auto mlir_module = mlir::parseSourceFile<mlir::ModuleOp>(
      tsl::GetDataDependencyFilepath(kBasicAttributesMlir), &mlir_context);

  AttributeEncoderRegistry attribute_encoder_registry;
  bc::Buffer buffer =
      EmitExecutable(attribute_encoder_registry, mlir_module.get()).value();

  bc::Executable executable(buffer.data());

  auto attributes = executable.attributes();

  ASSERT_EQ(attributes.size(), 15);

  auto attr_iter = attributes.begin();

  EXPECT_EQ(*attr_iter, "test string");
  ++attr_iter;

  EXPECT_EQ(*attr_iter, "ts");
  ++attr_iter;

  EXPECT_THAT(DecodeAttribute<int32_t>(*attr_iter), IsOkAndHolds(100));
  ++attr_iter;

  EXPECT_THAT(DecodeAttribute<int64_t>(*attr_iter), IsOkAndHolds(200));
  ++attr_iter;

  EXPECT_THAT(DecodeAttribute<float>(*attr_iter), IsOkAndHolds(FloatEq(3.0)));
  ++attr_iter;

  EXPECT_THAT(DecodeAttribute<uint8_t>(*attr_iter), IsOkAndHolds(0));
  ++attr_iter;

  bc::Vector<int64_t> list_of_i64((*attr_iter).data());
  EXPECT_THAT(list_of_i64, ElementsAreArray({0, 1, 2, 3, 4}));
  ++attr_iter;

  bc::Vector<int32_t> list_of_i32((*attr_iter).data());
  EXPECT_THAT(list_of_i32, ElementsAreArray({0, 1, 2, 3}));
  ++attr_iter;

  bc::Vector<bc::String> list_of_str((*attr_iter).data());
  EXPECT_THAT(list_of_str, ElementsAreArray({"string 0", "string 1"}));
  ++attr_iter;

  EXPECT_THAT(DecodeAttribute<uint32_t>(*attr_iter), IsOkAndHolds(1));
  EXPECT_EQ(executable.functions()[1].name().Get(), "callee");
  ++attr_iter;

  bc::Vector<int32_t> list_of_symbol_ref((*attr_iter).data());
  EXPECT_EQ(executable.functions()[2].name().Get(), "callee0");
  EXPECT_EQ(executable.functions()[3].name().Get(), "callee1");
  EXPECT_THAT(list_of_symbol_ref, ElementsAreArray({2, 3}));
  ++attr_iter;

  bc::Vector<int32_t> dense_array_of_i32((*attr_iter).data());
  EXPECT_THAT(dense_array_of_i32, ElementsAreArray({0, 1, 2}));
  ++attr_iter;

  bc::Vector<int64_t> dense_array_of_i64((*attr_iter).data());
  EXPECT_THAT(dense_array_of_i64, ElementsAreArray({0, 1, 2}));
  ++attr_iter;

  bc::Vector<int32_t> empty_dense_array((*attr_iter).data());
  EXPECT_TRUE(empty_dense_array.empty());
  ++attr_iter;

  bc::Vector<uint8_t> dense_array_of_bool((*attr_iter).data());
  EXPECT_THAT(dense_array_of_bool, ElementsAreArray({true, false}));

  auto kernels = executable.functions()[0].kernels();
  ASSERT_EQ(kernels.size(), 16);
  auto kernel_iter = kernels.begin();

  auto attribute_span = [&](auto kernel_iter) {
    return mlrt::AttributeSpan((*kernel_iter).attributes(), attributes);
  };

  EXPECT_EQ(attribute_span(kernel_iter).GetAs<bc::String>(0).Get(),
            "test string");
  ++kernel_iter;

  EXPECT_EQ(attribute_span(kernel_iter).GetAs<bc::String>(0).Get(), "ts");
  ++kernel_iter;

  EXPECT_EQ(attribute_span(kernel_iter).GetAs<int32_t>(0), 100);
  ++kernel_iter;

  EXPECT_EQ(attribute_span(kernel_iter).GetAs<int64_t>(0), 200);
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<float>(0), FloatEq(3.0));
  ++kernel_iter;

  EXPECT_EQ(attribute_span(kernel_iter).GetAs<uint8_t>(0), false);
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<int64_t>>(0),
              ElementsAreArray({0, 1, 2, 3, 4}));
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<int32_t>>(0),
              ElementsAreArray({0, 1, 2, 3}));
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<bc::String>>(0),
              ElementsAreArray({"string 0", "string 1"}));
  ++kernel_iter;

  EXPECT_EQ(attribute_span(kernel_iter).GetAs<uint32_t>(0), 1);
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<int32_t>>(0),
              ElementsAreArray({2, 3}));
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<int32_t>>(0),
              ElementsAreArray({0, 1, 2}));
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<int64_t>>(0),
              ElementsAreArray({0, 1, 2}));
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<int32_t>>(0),
              IsEmpty());
  ++kernel_iter;

  EXPECT_THAT(attribute_span(kernel_iter).GetAs<bc::Vector<bool>>(0),
              ElementsAreArray({true, false}));
}

TEST(MlirToByteCodeTest, UnsupportedAttributes) {
  constexpr char kUnsupportedAttributesMlir[] =
      "tensorflow/compiler/mlir/tfrt/translate/mlrt/testdata/"
      "unsupported_attributes.mlir";

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext mlir_context(registry);
  mlir_context.allowUnregisteredDialects();
  auto mlir_module = mlir::parseSourceFile<mlir::ModuleOp>(
      tsl::GetDataDependencyFilepath(kUnsupportedAttributesMlir),
      &mlir_context);

  AttributeEncoderRegistry attribute_encoder_registry;
  EXPECT_THAT(EmitExecutable(attribute_encoder_registry, mlir_module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Try to encode unsupported attribute: unit"));
}

class CustomDense {
 public:
  struct StorageType {
    using Self = StorageType;
    DEFINE_BYTECODE_FIELD(bc::Vector<int64_t>, shape);
    DEFINE_BYTECODE_FIELD(bc::Vector<uint32_t>, data);
  };

  class Constructor {
   public:
    Constructor(bc::Allocator* allocator, bc::BcAddr_t address)
        : allocator_(allocator), address_(address) {}

    template <typename... Args>
    auto construct_shape(Args&&... args) {
      return StorageType::construct_shape(allocator_, address_,
                                          std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto construct_data(Args&&... args) {
      return StorageType::construct_data(allocator_, address_,
                                         std::forward<Args>(args)...);
    }

    bc::BcAddr_t address() const { return address_; }

   private:
    bc::Allocator* allocator_;
    bc::BcAddr_t address_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit CustomDense(const char* p) : p_(p) {}

  bc::Vector<int64_t> shape() const { return StorageType::read_shape(p_); }
  bc::Vector<uint32_t> data() const { return StorageType::read_data(p_); }

 private:
  const char* p_ = nullptr;
};

absl::StatusOr<std::string> EncodeCustomDense(const ModuleEmitterContext&,
                                              mlir::Attribute attr) {
  auto dense_int_attr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr);
  if (!dense_int_attr)
    return absl::InvalidArgumentError(
        "The element of the custom dense attribute must be an integer.");

  if (mlir::cast<mlir::IntegerType>(dense_int_attr.getElementType())
          .getWidth() != 32) {
    return absl::InvalidArgumentError(
        "The element of the custom dense attribute must be an i32 integer.");
  }

  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);
  auto custom_dense_ctor = bc::New<CustomDense>(&allocator);

  auto shaped_type = dense_int_attr.getType();
  std::vector<int64_t> shape(shaped_type.getShape().begin(),
                             shaped_type.getShape().end());
  custom_dense_ctor.construct_shape(shape);

  custom_dense_ctor.construct_data(shaped_type.getNumElements())
      .Place(dense_int_attr.getRawData().data(),
             dense_int_attr.getRawData().size());

  return std::string(buffer.data(), buffer.size());
}

TEST(MlirToByteCodeTest, CustomDense) {
  constexpr char kCustomAttributesMlir[] =
      "tensorflow/compiler/mlir/tfrt/translate/mlrt/testdata/"
      "custom_attributes.mlir";

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext mlir_context(registry);
  mlir_context.allowUnregisteredDialects();
  auto mlir_module = mlir::parseSourceFile<mlir::ModuleOp>(
      tsl::GetDataDependencyFilepath(kCustomAttributesMlir), &mlir_context);

  AttributeEncoderRegistry attribute_encoder_registry;
  attribute_encoder_registry.Register("test_custom", &EncodeCustomDense);
  bc::Buffer buffer =
      EmitExecutable(attribute_encoder_registry, mlir_module.get()).value();

  bc::Executable executable(buffer.data());

  auto attributes = executable.attributes();

  ASSERT_EQ(attributes.size(), 10);
  for (int i = 0; i < 10; ++i) {
    bc::String attr_data = attributes[i];

    CustomDense custom_dense(attr_data.data());
    EXPECT_THAT(custom_dense.shape(), ElementsAreArray({1}));
    EXPECT_THAT(custom_dense.data(), ElementsAreArray({i}));
  }
}

}  // namespace
}  // namespace mlrt
