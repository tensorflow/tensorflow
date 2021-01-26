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
#include <cmath>
#include <fstream>

#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/unified_api_testutil.h"
#include "tensorflow/c/experimental/gradients/abstract_model.h"
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/test.h"

constexpr char TRAIN_IMAGES_PATH[] = "train-images-idx3-ubyte";
constexpr char TRAIN_LABELS_PATH[] = "train-labels-idx1-ubyte";
constexpr char TEST_IMAGES_PATH[] = "t10k-images-idx3-ubyte";
constexpr char TEST_LABELS_PATH[] = "t10k-labels-idx1-ubyte";

namespace tensorflow {
namespace gradients {
namespace internal {
namespace {

using tensorflow::TF_StatusPtr;

std::string GetMNISTPath(const std::string& sub_path) {
  static const char* dir_path =
      std::getenv("TF_CPP_GRADIENTS_MNIST_GRAD_TEST_DATASET_DIR");
  if (!dir_path) return sub_path;
  return dir_path + sub_path;
}

int ReverseInt(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = (i & 255);
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) +
         (static_cast<int>(c3) << 8) + c4;
}

void ReadMNISTHeader(std::ifstream* file, int* value) {
  file->read(reinterpret_cast<char*>(value), sizeof(*value));
  *value = ReverseInt(*value);
}

std::vector<std::vector<unsigned char>> ReadMNISTImage(
    const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return {};

  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;

  ReadMNISTHeader(&file, &magic_number);
  if (magic_number != 2051) return {};
  ReadMNISTHeader(&file, &number_of_images);
  ReadMNISTHeader(&file, &n_rows);
  ReadMNISTHeader(&file, &n_cols);

  std::vector<std::vector<unsigned char>> images(number_of_images);

  int image_size = n_cols * n_rows;
  for (int i = 0; i < number_of_images; i++) {
    images[i].resize(image_size);
    file.read(reinterpret_cast<char*>(images[i].data()), image_size);
  }

  return images;
}

std::vector<unsigned char> ReadMNISTLabel(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return {};

  int magic_number = 0;
  int number_of_labels = 0;

  ReadMNISTHeader(&file, &magic_number);
  if (magic_number != 2049) return {};
  ReadMNISTHeader(&file, &number_of_labels);

  std::vector<unsigned char> labels(number_of_labels);

  for (int i = 0; i < number_of_labels; i++) {
    file.read(reinterpret_cast<char*>(&labels[i]), 1);
  }

  return labels;
}

Status PreprocessMNISTImage(AbstractContext* ctx,
                            absl::Span<const std::vector<unsigned char>> images,
                            int batch_size,
                            std::vector<AbstractTensorHandle*>* outputs) {
  int number_of_images = images.size();
  int image_size = images[0].size();

  int64_t dims[] = {batch_size, image_size};
  int number_of_outputs =
      static_cast<int>(std::ceil(number_of_images * 1.0 / batch_size));
  std::unique_ptr<float[]> tensor_data(new float[image_size * batch_size]);
  int current_tensor_data_index = 0;
  outputs->resize(number_of_outputs);

  for (int i = 0; i < number_of_outputs; i++) {
    int current_batch_size =
        std::min(batch_size, number_of_images - i * batch_size);
    dims[0] = current_batch_size;

    for (int j = 0; j < current_batch_size; ++j) {
      for (int k = 0; k < image_size; ++k) {
        tensor_data[current_tensor_data_index++] =
            images[i * batch_size + j][k] / 255.0f;
      }
    }

    TF_RETURN_IF_ERROR(TestTensorHandleWithDims<float, TF_FLOAT>(
        ctx, tensor_data.get(), dims, 2, &((*outputs)[i])));
    current_tensor_data_index = 0;
  }
  return Status::OK();
}

Status PreprocessMNISTLabel(AbstractContext* ctx,
                            absl::Span<const unsigned char> labels,
                            int batch_size,
                            std::vector<AbstractTensorHandle*>* outputs) {
  int number_of_labels = labels.size();

  int64_t dims[] = {batch_size};
  int number_of_outputs =
      static_cast<int>(std::ceil(number_of_labels * 1.0 / batch_size));
  std::unique_ptr<int32_t[]> tensor_data(new int32_t[batch_size]);
  int current_tensor_data_index = 0;
  outputs->resize(number_of_outputs);

  for (int i = 0; i < number_of_outputs; i++) {
    int current_batch_size =
        std::min(batch_size, number_of_labels - i * batch_size);
    dims[0] = current_batch_size;

    for (int j = 0; j < current_batch_size; ++j) {
      tensor_data[current_tensor_data_index++] = labels[i * batch_size + j];
    }

    TF_RETURN_IF_ERROR(TestTensorHandleWithDims<int32_t, TF_INT32>(
        ctx, tensor_data.get(), dims, 1, &((*outputs)[i])));
    current_tensor_data_index = 0;
  }
  return Status::OK();
}

class SGD {
 public:
  SGD(float learning_rate) : learning_rate_(learning_rate) {}
  ~SGD() = default;

  Status operator()(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs) {
    std::vector<AbstractTensorHandle*> temp_outputs(1);
    std::string name_ops;

    AbstractTensorHandlePtr lr;
    {
      AbstractTensorHandle* lr_raw = nullptr;
      TF_RETURN_IF_ERROR(TestScalarTensorHandle<float, TF_FLOAT>(
          ctx, learning_rate_, &lr_raw));
      lr.reset(lr_raw);
    }

    for (size_t i{}; i < inputs.size(); ++i) {
      // `update = learning_rate * grad`
      name_ops = "UpdateWeights_Mul_" + std::to_string(i);
      TF_RETURN_IF_ERROR(ops::Mul(ctx, {lr.get(), inputs[i]},
                                  absl::MakeSpan(temp_outputs),
                                  name_ops.c_str()));
      AbstractTensorHandle* input = temp_outputs[0];

      // `grad = grad - update`
      name_ops = "UpdateWeights_Sub_" + std::to_string(i);
      TF_RETURN_IF_ERROR(ops::Sub(ctx, {outputs[i], input},
                                  absl::MakeSpan(temp_outputs),
                                  name_ops.c_str()));
      input->Unref();
      outputs[i] = temp_outputs[0];
    }
    return Status::OK();
  }

 private:
  float learning_rate_;
};

Status SparseCategoricalCrossentropy(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
    absl::Span<AbstractTensorHandle*> outputs) {
  // inputs = `[model_output, labels]`
  // outputs = loss_val
  std::vector<AbstractTensorHandle*> temp_outputs(2);
  TF_RETURN_IF_ERROR(ops::SparseSoftmaxCrossEntropyWithLogits(
      ctx, {inputs[0], inputs[1]}, absl::MakeSpan(temp_outputs),
      "SparseSoftmaxCrossEntropyWithLogits"));
  temp_outputs[1]->Unref();

  outputs[0] = temp_outputs[0];
  return Status::OK();
}

Status SparseCategoricalAccuracy(AbstractContext* ctx,
                                 absl::Span<AbstractTensorHandle* const> inputs,
                                 absl::Span<AbstractTensorHandle*> outputs) {
  std::vector<AbstractTensorHandle*> temp_outputs(1);
  TF_RETURN_IF_ERROR(
      ops::Equal(ctx, inputs[0], inputs[1], absl::MakeSpan(temp_outputs)));

  auto temp_input = temp_outputs[0];
  TF_RETURN_IF_ERROR(
      ops::Cast(ctx, temp_input, DT_FLOAT, absl::MakeSpan(temp_outputs)));
  temp_input->Unref();
  temp_input = temp_outputs[0];

  AbstractTensorHandlePtr axis;
  {
    AbstractTensorHandle* axis_raw = nullptr;
    TF_RETURN_IF_ERROR(
        TestScalarTensorHandle<int32_t, TF_INT32>(ctx, -1, &axis_raw));
    axis.reset(axis_raw);
  }
  TF_RETURN_IF_ERROR(
      ops::Mean(ctx, temp_input, axis.get(), absl::MakeSpan(temp_outputs)));
  temp_input->Unref();
  outputs[0] = temp_outputs[0];

  return Status::OK();
}

class MNISTModel : public AbstractModel {
 public:
  MNISTModel(AbstractContext* ctx, Model optimizer, GradientRegistry registry,
             bool use_function, int64_t image_size, Status* status)
      : AbstractModel(ctx, SparseCategoricalCrossentropy,
                      SparseCategoricalAccuracy, std::move(optimizer),
                      std::move(registry), use_function) {
    *status = InitializeWeights(ctx, image_size);
  }
  ~MNISTModel() = default;

  Status Compute(AbstractContext* ctx, AbstractTensorHandle* const input,
                 AbstractTensorHandle** output) override {
    // inputs = `[X, W1, Bias1, W2, Bias2]`

    std::vector<AbstractTensorHandle*> temp_outputs(1);
    // `mm_out_1 = tf.matmul(X,W1) + Bias1`
    TF_RETURN_IF_ERROR(ops::MatMul(ctx, {input, weights_[0]},
                                   absl::MakeSpan(temp_outputs),
                                   "MNIST_MatMul_1",
                                   /*transpose_a=*/false,
                                   /*transpose_b=*/false));
    auto temp_input = temp_outputs[0];
    TF_RETURN_IF_ERROR(ops::BiasAdd(ctx, {temp_input, weights_[1]},
                                    absl::MakeSpan(temp_outputs),
                                    "MNIST_BiasAdd_1"));
    temp_input->Unref();
    temp_input = temp_outputs[0];

    // `hidden_layer = tf.nn.relu(mm_out_1)`
    TF_RETURN_IF_ERROR(ops::Relu(ctx, {temp_input},
                                 absl::MakeSpan(temp_outputs), "MNIST_Relu"));
    temp_input->Unref();
    temp_input = temp_outputs[0];

    // `scores = tf.matmul(hidden_layer,W2)`
    TF_RETURN_IF_ERROR(
        ops::MatMul(ctx, {temp_input, weights_[2]},
                    absl::MakeSpan(temp_outputs), "MNIST_MatMul_2",
                    /*transpose_a=*/false, /*transpose_b=*/false));
    temp_input->Unref();
    temp_input = temp_outputs[0];
    TF_RETURN_IF_ERROR(ops::BiasAdd(ctx, {temp_input, weights_[3]},
                                    absl::MakeSpan(temp_outputs),
                                    "MNIST_BiasAdd_2"));

    *output = temp_outputs[0];
    return Status::OK();
  }

  Status operator()(AbstractContext* ctx, AbstractTensorHandle* const input,
                    AbstractTensorHandle** output) override {
    std::vector<AbstractTensorHandle*> temp_outputs(1);
    TF_RETURN_IF_ERROR(Compute(ctx, input, &temp_outputs[0]));
    AbstractTensorHandlePtr dimension;
    {
      AbstractTensorHandle* dimension_raw = nullptr;
      TF_RETURN_IF_ERROR(
          TestScalarTensorHandle<int32_t, TF_INT32>(ctx, -1, &dimension_raw));
      dimension.reset(dimension_raw);
    }
    auto temp_input = temp_outputs[0];
    TF_RETURN_IF_ERROR(ops::ArgMax(ctx, temp_input, dimension.get(),
                                   absl::MakeSpan(temp_outputs), DT_INT32));
    temp_input->Unref();
    *output = temp_outputs[0];
    return Status::OK();
  }

 private:
  Status InitializeWeights(AbstractContext* ctx, int64_t image_size) {
    weights_.resize(4);
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsRandom<float, TF_FLOAT>(
        ctx, -0.1, 0.1, {image_size, 128}, &(weights_[0])));
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsRandom<float, TF_FLOAT>(
        ctx, 0, 0, {128}, &(weights_[1])));
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsRandom<float, TF_FLOAT>(
        ctx, -0.1, 0.1, {128, 10}, &(weights_[2])));
    TF_RETURN_IF_ERROR(TestTensorHandleWithDimsRandom<float, TF_FLOAT>(
        ctx, 0, 0, {10}, &(weights_[3])));
    return Status::OK();
  }
};

class CppGradients
    : public ::testing::TestWithParam<std::tuple<const char*, bool, bool>> {
 protected:
  void SetUp() override {
    if (train_images_.empty() || train_labels_.empty() ||
        test_images_.empty() || test_labels_.empty()) {
      GTEST_SKIP() << "Failed to load MNIST dataset.";
    }

    TF_StatusPtr status(TF_NewStatus());
    TF_SetTracingImplementation(std::get<0>(GetParam()), status.get());
    status_ = StatusFromTF_Status(status.get());
    ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

    {
      AbstractContext* ctx_raw = nullptr;
      status_ =
          BuildImmediateExecutionContext(std::get<1>(GetParam()), &ctx_raw);
      ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
      immediate_execution_ctx_.reset(ctx_raw);
    }

    // Computing numerical gradients with TensorFloat-32 is numerically
    // unstable. Some forward pass tests also fail with TensorFloat-32 due to
    // low tolerances
    enable_tensor_float_32_execution(false);
  }

  AbstractContextPtr immediate_execution_ctx_;
  GradientRegistry registry_;
  Status status_;

  static std::vector<std::vector<unsigned char>> train_images_;
  static std::vector<unsigned char> train_labels_;
  static std::vector<std::vector<unsigned char>> test_images_;
  static std::vector<unsigned char> test_labels_;

 public:
  bool UseMlir() const { return strcmp(std::get<0>(GetParam()), "mlir") == 0; }
  bool UseFunction() const { return std::get<2>(GetParam()); }

  Status GetMNISTData(std::vector<AbstractTensorHandle*>* train_images,
                      std::vector<AbstractTensorHandle*>* train_labels,
                      std::vector<AbstractTensorHandle*>* test_images,
                      std::vector<AbstractTensorHandle*>* test_labels,
                      int train_batch_size, int test_batch_size) {
    TF_RETURN_IF_ERROR(PreprocessMNISTImage(immediate_execution_ctx_.get(),
                                            train_images_, train_batch_size,
                                            train_images));
    TF_RETURN_IF_ERROR(PreprocessMNISTLabel(immediate_execution_ctx_.get(),
                                            train_labels_, train_batch_size,
                                            train_labels));
    TF_RETURN_IF_ERROR(PreprocessMNISTImage(immediate_execution_ctx_.get(),
                                            test_images_, test_batch_size,
                                            test_images));
    TF_RETURN_IF_ERROR(PreprocessMNISTLabel(immediate_execution_ctx_.get(),
                                            test_labels_, test_batch_size,
                                            test_labels));
    return Status::OK();
  }
};

std::vector<std::vector<unsigned char>> CppGradients::train_images_ =
    ReadMNISTImage(GetMNISTPath(TRAIN_IMAGES_PATH));
std::vector<unsigned char> CppGradients::train_labels_ =
    ReadMNISTLabel(GetMNISTPath(TRAIN_LABELS_PATH));
std::vector<std::vector<unsigned char>> CppGradients::test_images_ =
    ReadMNISTImage(GetMNISTPath(TEST_IMAGES_PATH));
std::vector<unsigned char> CppGradients::test_labels_ =
    ReadMNISTLabel(GetMNISTPath(TEST_LABELS_PATH));

TEST_P(CppGradients, TestMNISTTraining) {
  if (UseFunction() || UseMlir()) {
    // TODO(b/168850692): Enable this.
    GTEST_SKIP() << "Can't take gradient of "
                    "SparseSoftmaxCrossEntropyWithLogits in tracing mode.\n"
                 << "BiasAdd SetAttrString has not been implemented yet.\n";
  }

  constexpr int batch_size = 128;
  constexpr int epoch = 6;
  constexpr float learning_rate = 0.01;
  int64_t image_size = train_images_[0].size();

  std::vector<AbstractTensorHandle*> train_images;
  std::vector<AbstractTensorHandle*> train_labels;
  std::vector<AbstractTensorHandle*> test_images;
  std::vector<AbstractTensorHandle*> test_labels;

  status_ = GetMNISTData(&train_images, &train_labels, &test_images,
                         &test_labels, batch_size, batch_size);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  // registry
  status_ = registry_.Register("BiasAdd", BiasAddRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  status_ = registry_.Register("MatMul", MatMulRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  status_ = registry_.Register("Relu", ReluRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  status_ = registry_.Register("SparseSoftmaxCrossEntropyWithLogits",
                               SparseSoftmaxCrossEntropyWithLogitsRegisterer);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  // initialize model
  auto sgd = SGD(learning_rate);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  auto mnist_model = MNISTModel(immediate_execution_ctx_.get(), std::move(sgd),
                                registry_, UseFunction(), image_size, &status_);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  // fit model
  status_ = mnist_model.Fit(train_images, train_labels, epoch);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  // evaluate
  std::vector<AbstractTensorHandle*> outputs(1);
  status_ =
      mnist_model.Evaluate(test_images, test_labels, absl::MakeSpan(outputs));
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();

  // accuracy should be greater than 90%.
  TF_Tensor* accuracy_tensor;
  status_ = GetValue(outputs[0], &accuracy_tensor);
  ASSERT_EQ(errors::OK, status_.code()) << status_.error_message();
  outputs[0]->Unref();

  float accuracy[1] = {0};
  memcpy(accuracy, TF_TensorData(accuracy_tensor),
         TF_TensorByteSize(accuracy_tensor));
  EXPECT_GE(accuracy[0], 0.90);

  TF_DeleteTensor(accuracy_tensor);
}

#ifdef PLATFORM_GOOGLE
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#else
INSTANTIATE_TEST_SUITE_P(
    UnifiedCAPI, CppGradients,
    ::testing::Combine(::testing::Values("graphdef", "mlir"),
                       /*tfrt*/ ::testing::Values(false),
                       /*use_function*/ ::testing::Values(true, false)));
#endif
}  // namespace
}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
