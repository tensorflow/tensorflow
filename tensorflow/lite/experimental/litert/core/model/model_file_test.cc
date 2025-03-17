// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// schema/mutable/schema_generated.h and schema/schema_generated.h (included
// through flatbuffer_tools.h via model.h) have the same #ifdef, thus this line
// need to be put at the top to ensure we get the "mutable" version.
#if 1
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#endif

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/core/dispatch_op_schema.h"
#include "tensorflow/lite/experimental/litert/core/model/buffer_manager.h"
#include "tensorflow/lite/experimental/litert/core/model/graph_validation.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_file_test_util.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/test_models.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"

namespace litert::internal {
namespace {

using ::litert::testing::GetTestFilePath;
using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::Values;
using ::testing::litert::IsError;

using ModelFactory = std::function<Expected<Model>()>;

static constexpr absl::string_view kAddSimple = "add_simple.tflite";
static constexpr absl::string_view kAddCst = "add_cst.tflite";
static constexpr absl::string_view kDynamicShapeModel =
    "dynamic_shape_tensor.tflite";
static constexpr absl::string_view kSimpleMultiOp = "simple_multi_op.tflite";
static constexpr absl::string_view kOneMul = "one_mul.tflite";
static constexpr absl::string_view kSimpleMultiSubgraph =
    "multi_subgraph.tflite";
static constexpr absl::string_view kCstMultiSubgraph =
    "cst_multi_subgraph.tflite";

// Load a model, then serialize and re-load. Used to test serialization.
Expected<Model> LoadModelThroughRoundTrip(absl::string_view filename) {
  auto model = Model::CreateFromFile(GetTestFilePath(filename));
  if (!model) {
    return model.Error();
  }

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  const auto opts = litert::SerializationOptions::Defaults();
  LITERT_RETURN_IF_ERROR(LiteRtSerializeModel(model->Release(), &data, &size,
                                              &offset, true, opts));

  // Reload model.
  LiteRtModel result = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtCreateModelFromBuffer(buf.Data(), buf.Size(), &result));

  return Model::CreateFromOwnedHandle(result);
}

ModelFactory MakeRoundTripFactory(absl::string_view filename) {
  return [=]() { return LoadModelThroughRoundTrip(filename); };
}

ModelFactory MakeLoadFactory(absl::string_view filename) {
  return [=]() { return Model::CreateFromFile(GetTestFilePath(filename)); };
}

// Test fixture parameterized by a file path to test model.
class TestWithModelPath : public ::testing::TestWithParam<absl::string_view> {
 protected:
  std::string GetTestModelPath() const {
    return testing::GetTestFilePath(GetParam());
  }
};

// Test fixture pareterized by a function that loads a model.
class TestWithModelFactory : public ::testing::TestWithParam<ModelFactory> {
 protected:
  Expected<Model> LoadModel() { return GetParam()(); }
};

// Simple tests
//===---------------------------------------------------------------------------

TEST(ModelLoadTest, BadFilepath) {
  LiteRtModel model = nullptr;
  EXPECT_THAT(LiteRtCreateModelFromFile("bad_path", &model),
              IsError(kLiteRtStatusErrorNotFound));
}

TEST(ModelLoadTest, BadFileData) {
  // NOLINTBEGIN
#ifndef NDEBUG
  // In debug mode, flatbuffers will `assert` while verifying. This will
  // cause this test to crash (as expected).
  GTEST_SKIP();
#endif
  std::filesystem::path test_file_path(::testing::TempDir());
  test_file_path.append("bad_file.txt");

  std::ofstream bad_file;
  bad_file.open(test_file_path.c_str());
  bad_file << "not_tflite";
  bad_file.close();

  LiteRtModel model = nullptr;
  EXPECT_THAT(LiteRtCreateModelFromFile(test_file_path.c_str(), &model),
              IsError(kLiteRtStatusErrorInvalidFlatbuffer));
  // NOLINTEND
}

TEST(ModelLoadTest, GetCustomOpCode) {
  auto model = litert::testing::LoadTestFileModel("simple_model_npu.tflite");
  ASSERT_TRUE(model);
  const auto& litert_model = *model.Get();
  const auto& op = *litert_model.MainSubgraph()->Ops().front();
  auto custom_op_code = GetCustomOpCode(litert_model, op);
  ASSERT_TRUE(custom_op_code.has_value());
  EXPECT_EQ(*custom_op_code, "DISPATCH_OP");
}

TEST(ModelLoadTest, WithMetadata) {
  constexpr static absl::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static absl::string_view kMetadataData = "My_Meta_Data";

  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();
  PushMetadata(kMetadataName, *tfl_model,
               BufferRef<uint8_t>(kMetadataData.data(), kMetadataData.size()));
  auto serialialized = SerializeFlatbuffer(*tfl_model);

  auto litert_model = LoadModelFromBuffer(serialialized);
  ASSERT_TRUE(litert_model);

  auto metadata = litert_model->get()->FindMetadata(kMetadataName);
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->StrView(), kMetadataData);
}

TEST(ModelSerializeTest, WithMetadata) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);

  constexpr static absl::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static absl::string_view kMetadataData = "My_Meta_Data";

  LITERT_ASSERT_OK(model.Get()->PushMetadata(
      kMetadataName, OwningBufferRef<uint8_t>(kMetadataData)));

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto metadata = re_loaded->get()->FindMetadata(kMetadataName);
  EXPECT_EQ(metadata->StrView(), kMetadataData);
}

TEST(ModelLoadTest, WithSignature) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  auto signature =
      litert_model.FindSignature(LiteRtSignatureT::kDefaultSignatureKey);
  ASSERT_TRUE(signature);

  EXPECT_EQ(signature->get().InputNames().size(), 1);
  EXPECT_EQ(signature->get().OutputNames().size(), 1);
  EXPECT_EQ(&signature->get().GetSubgraph(), litert_model.MainSubgraph());
}

TEST(ModelLoadTest, NoSignature) {
  auto model = *Model::CreateFromFile(testing::GetTfliteFilePath(
      "java/demo/app/src/main/assets/mobilenet_v1_1.0_224.tflite"));
  if (!model) {
    GTEST_SKIP() << "Model file is not available.";
  }
  auto& litert_model = *model.Get();
  auto signature =
      litert_model.FindSignature(LiteRtSignatureT::kDefaultSignatureKey);
  ASSERT_TRUE(signature);
  EXPECT_EQ(signature->get().InputNames().size(), 1);
  EXPECT_EQ(signature->get().OutputNames().size(), 1);
  EXPECT_EQ(&signature->get().GetSubgraph(), litert_model.MainSubgraph());
}

TEST(ModelSerializeTest, WithSignature) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  static constexpr char kInput[] = "foo";
  static constexpr char kOutput[] = "bar";
  static constexpr char kKey[] = "newKey";

  LiteRtSignatureT signature(litert_model.MainSubgraph(), {kInput}, {kOutput},
                             kKey);
  litert_model.EmplaceSignature(std::move(signature));

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto re_loaded_signature = re_loaded->get()->FindSignature(kKey);
  ASSERT_TRUE(re_loaded_signature);
  const auto& sig = re_loaded_signature->get();

  const auto& inputs = sig.InputNames();
  const auto& outputs = sig.OutputNames();
  EXPECT_THAT(inputs, ElementsAreArray({kInput}));
  EXPECT_THAT(outputs, ElementsAreArray({kOutput}));
  EXPECT_EQ(&sig.GetSubgraph(), re_loaded->get()->MainSubgraph());
}

TEST(ModelLoadTest, ReverseSignature) {
  auto model =
      litert::testing::LoadTestFileModel("reverse_signature_model.tflite");
  ASSERT_TRUE(model);
  auto& litert_model = *model.Get();

  auto signature = litert_model.FindSignature("serving_default");
  ASSERT_TRUE(signature);

  // Check if the input and output names are in the order of the subgraph
  // inputs and outputs instead of the signature appearance order.
  const auto& sig = signature->get();
  ASSERT_EQ(sig.InputNames().size(), 2);
  EXPECT_STREQ(sig.InputNames()[0].c_str(), "y");
  EXPECT_STREQ(sig.InputNames()[1].c_str(), "x");
  ASSERT_EQ(sig.OutputNames().size(), 2);
  EXPECT_STREQ(sig.OutputNames()[0].c_str(), "sum");
  EXPECT_STREQ(sig.OutputNames()[1].c_str(), "prod");

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto re_loaded_signature = re_loaded->get()->FindSignature("serving_default");
  ASSERT_TRUE(re_loaded_signature);

  // Check again with the serialized model.
  const auto& re_sig = re_loaded_signature->get();
  ASSERT_EQ(re_sig.InputNames().size(), 2);
  EXPECT_STREQ(re_sig.InputNames()[0].c_str(), "y");
  EXPECT_STREQ(re_sig.InputNames()[1].c_str(), "x");
  ASSERT_EQ(re_sig.OutputNames().size(), 2);
  EXPECT_STREQ(re_sig.OutputNames()[0].c_str(), "sum");
  EXPECT_STREQ(re_sig.OutputNames()[1].c_str(), "prod");
}

TEST(ModelLoadTest, WithOffsetTensorBuffer) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";

  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();
  const auto buf_ind = tfl_model->subgraphs[0]->tensors[0]->buffer;
  auto& tfl_buffer = tfl_model->buffers[buf_ind];
  tfl_buffer->offset = 1;
  tfl_buffer->size = 1;
  auto model_buf = SerializeFlatbuffer(*tfl_model);
  auto* packed_tfl = tflite::GetMutableModel(model_buf.Data());
  auto* buf = packed_tfl->mutable_buffers()->GetMutableObject(buf_ind);
  ASSERT_TRUE(buf->mutate_offset(model_buf.Size()));
  ASSERT_TRUE(buf->mutate_size(kTensorData.size()));
  OwningBufferRef<uint8_t> final_serializd(kTensorData.size() +
                                           model_buf.Size());
  std::memcpy(final_serializd.Data(), model_buf.Data(), model_buf.Size());
  std::memcpy(final_serializd.Data() + model_buf.Size(), kTensorData.data(),
              kTensorData.size());

  auto litert_model = LoadModelFromBuffer(final_serializd);
  ASSERT_TRUE(litert_model);

  const auto& weights_buffer =
      litert_model->get()->Subgraph(0).Tensor(0).Weights();
  EXPECT_EQ(weights_buffer.Buffer().StrView(), kTensorData);

  // The loaded buffer should indicate that it should be also serialized as
  // external.
  const auto will_append = weights_buffer.GetBufferManager()
                               ->GetContext(weights_buffer.GetBufferId())
                               ->get()
                               .should_append;
  EXPECT_TRUE(will_append);

  // All tensors in the first subgraph should have the same buffer manager as
  // the model.
  for (auto* tensor : litert_model->get()->Subgraph(0).Tensors()) {
    EXPECT_EQ(tensor->Weights().GetBufferManager(),
              litert_model->get()->Buffers());
  }
}

TEST(ModelSerializeTest, WithOffsetTensorBuffer) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& tensor = sg.EmplaceTensor();
  sg.EmplaceOp();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  OwningBufferRef<uint8_t> buffer(kTensorData);
  BufferContext context;
  context.should_append = true;
  SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify the op contains an offset and size to the byte code and the correct
  // name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();
  const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
  const auto tfl_buffer_ind = tfl_tensor->buffer;
  const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

  auto data =
      serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
  EXPECT_EQ(data, kTensorData);
}

TEST(ModelSerializeTest, WithMultipleOffsetTensorBuffer) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";
  static constexpr absl::string_view kTensorData2 = "SOME_TENSOR_DATA2";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  sg.EmplaceOp();

  {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
    auto& weights = tensor.Weights();
    weights.SetBufferManager(root.Buffers());

    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
    auto& weights = tensor.Weights();
    weights.SetBufferManager(root.Buffers());

    OwningBufferRef<uint8_t> buffer(kTensorData2);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify the op contains an offset and size to the byte code and the correct
  // name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData);
  }

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[1];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData2);
  }
}

TEST(ModelSerializeTest, WithSingleExternalBuffer) {
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "foo";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();

  OwningBufferRef<uint8_t> buffer(kByteCode);
  const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
  root.AttachAssetToOp(&op, buf_id, std::string(kName));

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify the op contains an offset and size to the byte code and the correct
  // name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();
  const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
  BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

  auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
  EXPECT_EQ(dispatch_opts.name, kName);
  EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                         dispatch_opts.bytecode_size),
            kByteCode);
}

TEST(ModelSerializeTest, WithMultipleUniqueExternalBuffer) {
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "foo";
  static constexpr absl::string_view kByteCode2 = "SOME_BYTE_CODE2";
  static constexpr absl::string_view kName2 = "bar";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& op2 = sg.EmplaceOp();

  OwningBufferRef<uint8_t> buffer(kByteCode);
  const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
  root.AttachAssetToOp(&op, buf_id, std::string(kName));

  OwningBufferRef<uint8_t> buffer2(kByteCode2);
  const auto buf_id2 = root.Buffers()->RegisterOwnedBuffer(std::move(buffer2));
  root.AttachAssetToOp(&op2, buf_id2, std::string(kName2));

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify both ops contains an offset and size to the byte code and the
  // correct name.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[1]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName2);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode2);
  }
}

TEST(ModelSerializeTest, WithSharedExternalBuffer) {
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "foo";
  static constexpr absl::string_view kName2 = "bar";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& op2 = sg.EmplaceOp();

  OwningBufferRef<uint8_t> buffer(kByteCode);
  const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));

  root.AttachAssetToOp(&op, buf_id, std::string(kName));
  root.AttachAssetToOp(&op2, buf_id, std::string(kName2));

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  // Verify both ops point to the same appended buffer.
  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);

  auto tfl = fb->get()->Unpack();

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[1]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName2);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }
}

TEST(ModelSerializeTest, WithOffsetTensorBufferAndOpAsset) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "name";

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& tensor = sg.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  {
    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  {
    OwningBufferRef<uint8_t> buffer(kByteCode);
    const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
    root.AttachAssetToOp(&op, buf_id, std::string(kName));
  }

  auto serialized = SerializeModel(std::move(root));
  ASSERT_TRUE(serialized);

  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);
  auto tfl = fb->get()->Unpack();

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }
}

TEST(ModelSerializeTest, WithOffsetTensorBufferAndOpAssetHasAlignment) {
  static constexpr absl::string_view kTensorData = "SOME_TENSOR_DATA";
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr absl::string_view kName = "name";
  static constexpr size_t kAlignment = 32;

  LiteRtModelT root;
  auto& sg = root.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  auto& tensor = sg.EmplaceTensor();
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {}));
  auto& weights = tensor.Weights();
  weights.SetBufferManager(root.Buffers());

  {
    OwningBufferRef<uint8_t> buffer(kTensorData);
    BufferContext context;
    context.should_append = true;
    SetWeightsFromOwnedBuffer(weights, std::move(buffer), context);
  }

  {
    OwningBufferRef<uint8_t> buffer(kByteCode);
    const auto buf_id = root.Buffers()->RegisterOwnedBuffer(std::move(buffer));
    root.AttachAssetToOp(&op, buf_id, std::string(kName));
  }

  auto serialized = SerializeModel(std::move(root), kAlignment);
  ASSERT_TRUE(serialized);

  auto fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(fb);
  auto tfl = fb->get()->Unpack();

  {
    const auto& tfl_tensor = tfl->subgraphs[0]->tensors[0];
    const auto tfl_buffer_ind = tfl_tensor->buffer;
    const auto& tfl_buffer = tfl->buffers[tfl_buffer_ind];

    auto data =
        serialized->StrView().substr(tfl_buffer->offset, tfl_buffer->size);
    EXPECT_EQ(data, kTensorData);
  }

  {
    const auto& opts = tfl->subgraphs[0]->operators[0]->custom_options;
    BufferRef<uint8_t> opts_buffer(opts.data(), opts.size());

    auto dispatch_opts = GetDispatchOpOptions(opts_buffer);
    EXPECT_EQ(dispatch_opts.name, kName);
    ASSERT_EQ(dispatch_opts.bytecode_offset % kAlignment, 0);
    EXPECT_EQ(serialized->StrView().substr(dispatch_opts.bytecode_offset,
                                           dispatch_opts.bytecode_size),
              kByteCode);
  }
}

// Tests that explicitly check litert graph structure.
//===---------------------------------------------------------------------------

using AddSimpleTest = TestWithModelFactory;

TEST_P(AddSimpleTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func(arg0)
  //  output = tfl.add(arg0, arg0)
  //  return(output)
  //

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

  ASSERT_EQ(ops.size(), 1);
  const auto& op = ops.front();

  const TensorTypeInfo float_2by2_type(ElementType::Float32, {2, 2});
  ASSERT_TRUE(
      MatchOpType(op, {float_2by2_type, float_2by2_type}, {float_2by2_type}));
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);

  const auto op_inputs = op.Inputs();
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs.front().Get(), subgraph_inputs.front().Get());
  ASSERT_EQ(op_inputs.front().Get(), op_inputs.back().Get());

  const auto op_outputs = op.Outputs();
  ASSERT_EQ(op_outputs.size(), 1);
  ASSERT_EQ(op_outputs.front().Get(), subgraph_outputs.front().Get());

  ASSERT_FALSE(subgraph_outputs.front().IsConstant());
  ASSERT_FALSE(subgraph_inputs.front().IsConstant());
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, AddSimpleTest,
                         Values(MakeLoadFactory(kAddSimple)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, AddSimpleTest,
                         Values(MakeRoundTripFactory(kAddSimple)));

using AddCstTest = TestWithModelFactory;

TEST_P(AddCstTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func(arg0)
  //  cst = ConstantTensor([1, 2, 3, 4])
  //  output = tfl.add(arg0, cst)
  //  return(output)
  //

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

  ASSERT_EQ(ops.size(), 1);
  const auto& op = ops.front();

  const TensorTypeInfo float_by4_type(ElementType::Float32, {4});
  ASSERT_TRUE(
      MatchOpType(op, {float_by4_type, float_by4_type}, {float_by4_type}));
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);

  const auto op_inputs = op.Inputs();
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs.front().Get(), subgraph_inputs.front().Get());
  ASSERT_TRUE(MatchWeights(op_inputs.back(),
                           absl::Span<const float>({1.0, 2.0, 3.0, 4.0})));

  const auto op_outputs = op.Outputs();
  ASSERT_EQ(op_outputs.size(), 1);
  ASSERT_EQ(op_outputs.front().Get(), subgraph_outputs.front().Get());

  ASSERT_FALSE(subgraph_outputs.front().IsConstant());
  ASSERT_FALSE(subgraph_inputs.front().IsConstant());
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, AddCstTest,
                         Values(MakeLoadFactory(kAddCst)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, AddCstTest,
                         Values(MakeRoundTripFactory(kAddCst)));

using SimpleMultiOpTest = TestWithModelFactory;

TEST_P(SimpleMultiOpTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

  ASSERT_EQ(ops.size(), 4);

  for (const auto& op : ops) {
    const auto inputs = op.Inputs();
    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(inputs.front().Get(), inputs.back().Get());
  }

  const TensorTypeInfo float_2by2_type(ElementType::Float32, {2, 2});

  ASSERT_TRUE(MatchOpType(ops.at(2), {float_2by2_type, float_2by2_type},
                          {float_2by2_type}));
  EXPECT_EQ(ops.at(2).Code(), kLiteRtOpCodeTflMul);
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, SimpleMultiOpTest,
                         Values(MakeLoadFactory(kSimpleMultiOp)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, SimpleMultiOpTest,
                         Values(MakeRoundTripFactory(kSimpleMultiOp)));

using SimpleMultiSubgraphTest = TestWithModelFactory;

TEST_P(SimpleMultiSubgraphTest, CheckGraph) {
  auto model_wrap = LoadModel();
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap->Get();

  ASSERT_EQ(model.NumSubgraphs(), 3);

  {
    auto& main = *model.MainSubgraph();
    EXPECT_EQ(main.NumInputs(), 1);
    EXPECT_EQ(main.NumOutputs(), 1);
    EXPECT_EQ(main.Ops().size(), 1);
    EXPECT_EQ(main.Tensors().size(), 3);
    auto& op = main.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(-1.0)));
    EXPECT_TRUE(ValidateLocalTopology(main.Ops().cbegin(), main.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(main));
  }

  {
    auto& func1 = model.Subgraph(1);
    EXPECT_EQ(func1.NumInputs(), 1);
    EXPECT_EQ(func1.NumOutputs(), 1);
    EXPECT_EQ(func1.Ops().size(), 1);
    EXPECT_EQ(func1.Tensors().size(), 3);
    auto& op = func1.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(1.0)));
    EXPECT_TRUE(
        ValidateLocalTopology(func1.Ops().cbegin(), func1.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(func1));
  }

  {
    auto& func2 = model.Subgraph(2);
    EXPECT_EQ(func2.NumInputs(), 1);
    EXPECT_EQ(func2.NumOutputs(), 1);
    EXPECT_EQ(func2.Ops().size(), 1);
    EXPECT_EQ(func2.Tensors().size(), 3);
    auto& op = func2.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(2.0)));
    EXPECT_TRUE(
        ValidateLocalTopology(func2.Ops().cbegin(), func2.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(func2));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, SimpleMultiSubgraphTest,
                         Values(MakeLoadFactory(kSimpleMultiSubgraph)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, SimpleMultiSubgraphTest,
                         Values(MakeRoundTripFactory(kSimpleMultiSubgraph)));

// Test when flatbuffer export has optimized multiple tensors to share the
// same buffer.
using MultiSubgraphDupeConstTest = TestWithModelFactory;

TEST_P(MultiSubgraphDupeConstTest, CheckGraph) {
  static constexpr std::array kWeights = {1.0, 2.0, 3.0, 4.0};

  auto model_wrap = LoadModel();
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap->Get();

  ASSERT_EQ(model.NumSubgraphs(), 2);

  {
    ASSERT_EQ(model.Subgraph(0).Ops().size(), 1);
    ASSERT_EQ(model.Subgraph(0).Tensors().size(), 3);
    auto& cst = model.Subgraph(0).Op(0).Input(1);
    Tensor t(&cst);
    EXPECT_THAT(*t.WeightsData<float>(), ElementsAreArray(kWeights));
  }

  {
    ASSERT_EQ(model.Subgraph(1).Ops().size(), 1);
    ASSERT_EQ(model.Subgraph(1).Tensors().size(), 3);
    auto& cst = model.Subgraph(1).Op(0).Input(1);
    Tensor t(&cst);
    EXPECT_THAT(*t.WeightsData<float>(), ElementsAreArray(kWeights));
  }
  auto buf_id_0 = model.Subgraph(0).Op(0).Input(1).Weights().GetBufferId();
  auto buf_id_1 = model.Subgraph(1).Op(0).Input(1).Weights().GetBufferId();
  ASSERT_EQ(buf_id_0, buf_id_1);
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, MultiSubgraphDupeConstTest,
                         Values(MakeLoadFactory(kCstMultiSubgraph)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, MultiSubgraphDupeConstTest,
                         Values(MakeRoundTripFactory(kCstMultiSubgraph)));

// Tests that programmatically check litert against tflite models.
//===---------------------------------------------------------------------------

using ModelLoadOpCheckTest = TestWithModelPath;

TEST_P(ModelLoadOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(flatbuffer);
  auto expected_fb = flatbuffer->get()->Unpack();

  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);

  const auto* subgraph = model->get()->MainSubgraph();
  const auto& ops = subgraph->Ops();

  const auto& fb_subgraph = *expected_fb->subgraphs.front();
  const auto& fb_ops = fb_subgraph.operators;
  const auto& fb_tensors = fb_subgraph.tensors;

  ASSERT_EQ(ops.size(), fb_ops.size());

  auto get_tfl_tensor = [&](uint32_t ind) -> const TflTensor& {
    return *fb_tensors.at(ind);
  };

  for (auto i = 0; i < ops.size(); ++i) {
    ASSERT_TRUE(EqualsFbOp(*ops.at(i), *fb_ops.at(i), get_tfl_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadQuantizedOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

INSTANTIATE_TEST_SUITE_P(ModelLoadDynamicOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn({kDynamicShapeModel}));

using ModelSerializeOpCheckTest = TestWithModelPath;

TEST_P(ModelSerializeOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  // Save the initial fb for comparison.
  auto expected_fb_data = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(expected_fb_data);
  auto expected_fb = expected_fb_data->get()->Unpack();

  // Round trip the model.
  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);
  auto serialized = SerializeModel(std::move(**model));

  auto actual_fb_data = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(actual_fb_data);
  auto actual_fb = actual_fb_data->get()->Unpack();

  const auto& expected_fb_subgraph = *expected_fb->subgraphs.front();
  const auto& expected_fb_ops = expected_fb_subgraph.operators;
  const auto& expected_fb_tensors = expected_fb_subgraph.tensors;

  const auto& actual_fb_subgraph = *actual_fb->subgraphs.front();
  const auto& actual_fb_ops = actual_fb_subgraph.operators;
  const auto& actual_fb_tensors = actual_fb_subgraph.tensors;

  ASSERT_EQ(expected_fb_ops.size(), actual_fb_ops.size());
  for (auto i = 0; i < actual_fb_ops.size(); ++i) {
    const auto& expected = *expected_fb_ops.at(i);
    const auto& actual = *actual_fb_ops.at(i);
    EXPECT_EQ(expected.inputs.size(), actual.inputs.size());
    EXPECT_EQ(expected.outputs.size(), actual.outputs.size());
  }

  ASSERT_EQ(expected_fb_tensors.size(), actual_fb_tensors.size());
  for (auto i = 0; i < actual_fb_tensors.size(); ++i) {
    const auto& expected = *expected_fb_tensors.at(i);
    const auto& actual = *actual_fb_tensors.at(i);

    EXPECT_EQ(actual.type, expected.type);
    EXPECT_EQ(actual.shape, expected.shape);
    EXPECT_EQ(actual.shape_signature, expected.shape_signature);

    const auto expected_q_params = expected.quantization.get();
    const auto actual_q_params = actual.quantization.get();

    const auto neither_quantized =
        !IsQuantized(expected_q_params) && !IsQuantized(actual_q_params);
    const auto both_per_tensor = IsPerTensorQuantized(expected_q_params) &&
                                 IsPerTensorQuantized(actual_q_params);
    ASSERT_TRUE(neither_quantized || both_per_tensor);

    if (both_per_tensor) {
      const auto expected_per_tensor = AsPerTensorQparams(expected_q_params);
      const auto actual_per_tensor = AsPerTensorQparams(actual_q_params);
      EXPECT_EQ(*expected_per_tensor, *actual_per_tensor);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ModelSerializeOpCheckTest, ModelSerializeOpCheckTest,
                         ::testing::ValuesIn({kOneMul, kDynamicShapeModel}));

INSTANTIATE_TEST_SUITE_P(ModelSerializeQuantizedOpCheckTest,
                         ModelSerializeOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

}  // namespace
}  // namespace litert::internal
