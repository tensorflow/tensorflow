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
#include "tensorflow/c/experimental/tensor_list/tensor_list.h"

#include "absl/container/inlined_vector.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/tensor_list.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
TF_Tensor* TF_TensorFromTensor(const Tensor& src, Status* status);
Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);

TEST(TensorList, CreateAndDelete) {
  TF_TensorList* list = TF_NewTensorList();
  ASSERT_NE(nullptr, list);
  TF_DeleteTensorList(list);
}

TEST(TensorList, PushBackToEmptyList) {
  TF_TensorList* list = TF_NewTensorList();
  ASSERT_NE(nullptr, list);
  const int num_bytes = 6 * sizeof(float);
  int64_t dims[] = {2, 3};
  TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, num_bytes);
  TF_TensorListPush(list, tensor);

  int size = TF_TensorListSize(list);
  EXPECT_EQ(1, size);

  TF_DeleteTensor(tensor);
  TF_DeleteTensorList(list);
}

TEST(TensorList, PushAndPopList) {
  TF_TensorList* list = TF_NewTensorList();
  ASSERT_NE(nullptr, list);

  std::vector<TF_Tensor*> tensors;
  int tensors_size = 10;
  for (int i = 0; i < tensors_size; i++) {
    const int num_bytes = 6 * sizeof(float);
    int64_t dims[] = {2, 3};
    TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, num_bytes);
    TF_TensorListPush(list, tensor);
    tensors.push_back(tensor);
  }

  int size = TF_TensorListSize(list);
  EXPECT_EQ(tensors_size, size);

  TF_TensorListPop(list);
  size = TF_TensorListSize(list);
  EXPECT_EQ(tensors_size - 1, size);

  for (int i = 0; i < tensors_size; i++) {
    TF_DeleteTensor(tensors[i]);
  }
  TF_DeleteTensorList(list);
}

TEST(TensorList, SetItemToList) {
  TF_TensorList* list = TF_NewTensorList();
  ASSERT_NE(nullptr, list);

  TF_Status* status = TF_NewStatus();

  const int num_bytes = 6 * sizeof(float);
  int64_t dims[] = {2, 3};
  TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, num_bytes);
  TF_TensorListPush(list, tensor);

  int64_t dims2[] = {1, 8};
  TF_Tensor* tensor2 = TF_AllocateTensor(TF_FLOAT, dims2, 2, 8 * sizeof(float));
  TF_TensorListSetTensor(list, 0, tensor2, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  TF_TensorListSetTensor(list, 1, tensor2, status);
  EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(status));

  TF_Tensor* t = nullptr;
  TF_TensorListGetTensor(list, 1, &t, status);
  EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(status));
  EXPECT_EQ(nullptr, t);

  TF_TensorListGetTensor(list, 0, &t, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  EXPECT_NE(nullptr, t);

  int64_t dim1 = TF_Dim(t, 0);
  int64_t dim2 = TF_Dim(t, 1);
  EXPECT_EQ(dim1, 1);
  EXPECT_EQ(dim2, 8);

  TF_DeleteTensor(tensor);
  TF_DeleteTensor(tensor2);
  TF_DeleteTensor(t);

  TF_DeleteStatus(status);

  TF_DeleteTensorList(list);
}

TEST(TensorList, TensorListMetaInfo) {
  TF_TensorList* list = TF_NewTensorList();
  ASSERT_NE(nullptr, list);

  const int num_bytes = 6 * sizeof(float);
  int64_t dims[] = {2, 3};
  TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, 2, num_bytes);
  TF_TensorListPush(list, tensor);

  TF_TensorListSetDataType(list, TF_FLOAT);
  TF_TensorListSetShape(list, 2, dims);

  {
    int size = TF_TensorListSize(list);
    EXPECT_EQ(size, 1);

    int64_t num_dims = TF_TensorListNumDims(list);
    EXPECT_EQ(num_dims, 2);

    int64_t dim1 = TF_TensorListDim(list, 0);
    int64_t dim2 = TF_TensorListDim(list, 1);
    EXPECT_EQ(dim1, 2);
    EXPECT_EQ(dim2, 3);

    TF_DataType dtype = TF_TensorListGetDataType(list);
    EXPECT_EQ(dtype, TF_FLOAT);
  }

  TF_DeleteTensor(tensor);
  TF_DeleteTensorList(list);
}

TEST(TensorList, GetTensorList) {
  ::tensorflow::TensorList list;
  ::tensorflow::Tensor tensor =
      ::tensorflow::Tensor(::tensorflow::DT_FLOAT, TensorShape({2, 3}));
  list.tensors().push_back(tensor);

  ::tensorflow::Tensor t = ::tensorflow::Tensor(DT_VARIANT, TensorShape({}));
  t.scalar<::tensorflow::Variant>()() = list;
  ::tensorflow::Status s;
  TF_Tensor* tf_t = TF_TensorFromTensor(t, &s);

  TF_TensorList* tf_list = nullptr;
  TF_Status* status = TF_NewStatus();

  TF_GetTensorListFromTensor(tf_t, 1, &tf_list, status);
  EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(status));
  EXPECT_EQ(tf_list, nullptr);

  TF_GetTensorListFromTensor(tf_t, 0, &tf_list, status);
  EXPECT_NE(nullptr, tf_list);
  EXPECT_EQ(TF_OK, TF_GetCode(status));

  int size = TF_TensorListSize(tf_list);
  EXPECT_EQ(1, size);

  TF_DeleteStatus(status);
  TF_DeleteTensor(tf_t);
}

TEST(TensorList, SetTensorList) {
  ::tensorflow::Tensor t = ::tensorflow::Tensor(DT_VARIANT, TensorShape({}));
  ::tensorflow::Status s;
  TF_Tensor* tf_t = TF_TensorFromTensor(t, &s);

  TF_Status* status = TF_NewStatus();
  TF_TensorList* list = TF_NewTensorList();

  TF_SetTensorListToTensor(list, 1, tf_t, status);
  EXPECT_EQ(TF_OUT_OF_RANGE, TF_GetCode(status));

  TF_SetTensorListToTensor(list, 0, tf_t, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  EXPECT_NE(nullptr, tf_t);

  TF_TensorList* get_back = nullptr;
  TF_GetTensorListFromTensor(tf_t, 0, &get_back, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  EXPECT_NE(nullptr, get_back);

  EXPECT_EQ(1, TF_TensorElementCount(tf_t));

  TF_DeleteStatus(status);
  TF_DeleteTensorList(list);
  TF_DeleteTensor(tf_t);
}

TEST(TensorList, CopyTensorList) {
  ::tensorflow::Status s;
  TF_Status* status = TF_NewStatus();
  TF_TensorList* list = TF_NewTensorList();

  {
    Tensor t = Tensor(DT_FLOAT, TensorShape({2, 3}));
    TF_Tensor* tf_t = TF_TensorFromTensor(t, &s);
    TF_TensorListPush(list, tf_t);
    TF_DeleteTensor(tf_t);
  }

  TF_TensorList* copy = TF_NewTensorList();
  TF_TensorListCopy(list, copy);

  EXPECT_EQ(1, TF_TensorListSize(copy));
  EXPECT_EQ(1, TF_TensorListSize(list));

  Tensor t = Tensor(DT_FLOAT, TensorShape({2, 3}));
  TF_Tensor* tf_t = TF_TensorFromTensor(t, &s);
  TF_TensorListPush(copy, tf_t);
  EXPECT_EQ(2, TF_TensorListSize(copy));
  EXPECT_EQ(1, TF_TensorListSize(list));

  TF_DeleteStatus(status);
  TF_DeleteTensorList(list);
  TF_DeleteTensorList(copy);
  TF_DeleteTensor(tf_t);
}

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
    return cpu_allocator();
  }
};

static std::unique_ptr<OpKernel> GetFakeKernel(const char* device_name,
                                               const char* op_name,
                                               const char* node_name,
                                               Status* status) {
  NodeDef def;
  def.set_op(op_name);
  def.set_name(node_name);
  def.set_device(device_name);
  def.add_input("input1");
  def.add_input("input2");

  AttrValue v;
  v.set_type(DataType::DT_FLOAT);
  (*def.mutable_attr())["SomeDataTypeAttr"] = v;

  return CreateOpKernel(DeviceType(device_name), nullptr, nullptr, def, 1,
                        status);
}

TEST(TensorList, ForwardInputOrCreateNewList) {
  const char* node_name = "ForwardInputOrCreateNewList";
  const char* op_name = "FakeListOp";
  const char* device_name = "FakeDeviceName";

  REGISTER_OP(op_name)
      .Input("input1: variant")
      .Input("input2: float")
      .Output("output1: variant")
      .Attr("SomeDataTypeAttr: type");

  // A kernel whose Compute function that forwards a tensorlist input to output
  auto my_compute_func = [](void* kernel, TF_OpKernelContext* ctx) {
    TF_Status* s = TF_NewStatus();
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, s);
    TF_TensorList* input_list = nullptr;
    TF_TensorList* output_list = nullptr;
    TF_GetTensorListFromTensor(input, 0, &input_list, s);
    TF_ForwardInputOrCreateNewList(ctx, 0, 0, input_list, &output_list, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s));
    EXPECT_EQ(TF_TensorListSize(input_list), TF_TensorListSize(output_list));
    TF_DeleteStatus(s);
    TF_DeleteTensor(input);
  };

  TF_KernelBuilder* builder = TF_NewKernelBuilder(op_name, device_name, nullptr,
                                                  my_compute_func, nullptr);

  {
    TF_Status* status = TF_NewStatus();
    TF_RegisterKernelBuilder(node_name, builder, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status));
    TF_DeleteStatus(status);
  }

  {
    OpKernelContext::Params p;
    DummyDevice dummy_device(nullptr);
    p.device = &dummy_device;
    AllocatorAttributes alloc_attrs;
    p.output_attr_array = &alloc_attrs;

    TensorList list;
    list.element_dtype = DT_FLOAT;
    list.element_shape = PartialTensorShape({2, 3});
    list.tensors().push_back(Tensor(DT_FLOAT, TensorShape({2, 3})));

    Tensor t = Tensor(DT_VARIANT, {});
    t.scalar<Variant>()() = list;

    gtl::InlinedVector<TensorValue, 4> inputs;
    inputs.emplace_back(&t);
    p.inputs = &inputs;

    Status status;
    std::unique_ptr<OpKernel> kernel =
        GetFakeKernel(device_name, op_name, node_name, &status);
    EXPECT_EQ(Status::OK(), status);
    ASSERT_NE(nullptr, kernel.get());

    p.op_kernel = kernel.get();
    OpKernelContext ctx(&p);
    kernel->Compute(&ctx);

    TensorList* output_list =
        ctx.mutable_output(0)->scalar<Variant>()().get<TensorList>();
    ASSERT_EQ(1, output_list->tensors().size());
  }
}
}  // namespace tensorflow
