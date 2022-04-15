/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/resource_handle.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

namespace {
std::string DtypeAndShapesToString(
    const std::vector<DtypeAndPartialTensorShape>& dtype_and_shapes) {
  std::vector<std::string> dtype_and_shape_strings;
  dtype_and_shape_strings.reserve(dtype_and_shapes.size());
  for (const DtypeAndPartialTensorShape& dtype_and_shape : dtype_and_shapes) {
    // Note that it is a bit unfortunate to return int/enum as dtype, given we
    // can't directly use DataTypeString due to circular dependency.
    dtype_and_shape_strings.push_back(
        absl::StrFormat("DType enum: %d, Shape: %s", dtype_and_shape.dtype,
                        dtype_and_shape.shape.DebugString()));
  }
  return absl::StrFormat("[ %s ]", absl::StrJoin(dtype_and_shape_strings, ","));
}
}  // namespace

// Must be declared here for pre-C++17 compatibility.
/* static */ constexpr const char* ResourceHandle::ANONYMOUS_NAME;

ResourceHandle::ResourceHandle() {}

ResourceHandle::ResourceHandle(const ResourceHandleProto& proto) {
  TF_CHECK_OK(FromProto(proto));
}

Status ResourceHandle::BuildResourceHandle(const ResourceHandleProto& proto,
                                           ResourceHandle* out) {
  if (out == nullptr)
    return errors::Internal(
        "BuildResourceHandle() was called with nullptr for the output");
  return out->FromProto(proto);
}

ResourceHandle::~ResourceHandle() {}

void ResourceHandle::AsProto(ResourceHandleProto* proto) const {
  proto->set_device(device());
  proto->set_container(container());
  proto->set_name(name());
  proto->set_hash_code(hash_code());
  proto->set_maybe_type_name(maybe_type_name());
  for (const auto& dtype_and_shape_pair : dtypes_and_shapes_) {
    auto dtype_and_shape = proto->add_dtypes_and_shapes();
    dtype_and_shape->set_dtype(dtype_and_shape_pair.dtype);
    dtype_and_shape_pair.shape.AsProto(dtype_and_shape->mutable_shape());
  }
}

Status ResourceHandle::FromProto(const ResourceHandleProto& proto) {
  set_device(proto.device());
  set_container(proto.container());
  set_name(proto.name());
  set_hash_code(proto.hash_code());
  set_maybe_type_name(proto.maybe_type_name());
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
  for (const auto& dtype_and_shape : proto.dtypes_and_shapes()) {
    DataType dtype = dtype_and_shape.dtype();
    PartialTensorShape shape;
    Status s = PartialTensorShape::BuildPartialTensorShape(
        dtype_and_shape.shape(), &shape);
    if (!s.ok()) {
      return s;
    }
    dtypes_and_shapes.push_back(DtypeAndPartialTensorShape{dtype, shape});
  }
  dtypes_and_shapes_ = std::move(dtypes_and_shapes);
  return Status::OK();
}

string ResourceHandle::SerializeAsString() const {
  ResourceHandleProto proto;
  AsProto(&proto);
  return proto.SerializeAsString();
}

bool ResourceHandle::ParseFromString(const string& s) {
  ResourceHandleProto proto;
  return proto.ParseFromString(s) && FromProto(proto).ok();
}

string ResourceHandle::DebugString() const {
  return absl::StrFormat(
      "device: %s container: %s name: %s hash_code: 0x%X maybe_type_name %s, "
      "dtype and shapes : %s",
      device(), container(), name(), hash_code(),
      port::Demangle(maybe_type_name()),
      DtypeAndShapesToString(dtypes_and_shapes()));
}
string ResourceHandle::SummarizeValue() const {
  return absl::StrFormat(
      "ResourceHandle(name=\"%s\", device=\"%s\", container=\"%s\", "
      "type=\"%s\", dtype and shapes : \"%s\")",
      name(), device(), container(), port::Demangle(maybe_type_name()),
      DtypeAndShapesToString(dtypes_and_shapes()));
}

ResourceHandle ResourceHandle::MakeRefCountingHandle(
    ResourceBase* resource, const string& device_name,
    const TypeIndex& type_index,
    const std::vector<DtypeAndPartialTensorShape>& dtypes_and_shapes,
    const absl::optional<ManagedStackTrace>& definition_stack_trace) {
  ResourceHandle result;
  result.resource_.reset(resource, /*add_ref=*/false);
  result.set_device(device_name);
  // All resources owned by anonymous handles are put into the same container,
  // and they get process-unique handle names.
  result.set_container("Anonymous");
  result.set_definition_stack_trace(definition_stack_trace);
  result.set_name(
      absl::StrFormat("Resource-%d-at-%p", GenerateUniqueId(), resource));
  result.set_hash_code(type_index.hash_code());
  result.set_maybe_type_name(type_index.name());
  result.set_dtypes_and_shapes(dtypes_and_shapes);
  return result;
}

Status ResourceHandle::ValidateType(const TypeIndex& type_index) const {
  if (type_index.hash_code() != hash_code()) {
    return errors::InvalidArgument(
        "Trying to access a handle's resource using the wrong type. ",
        "The handle points to a resource (name '", name(), "') of type '",
        port::Demangle(maybe_type_name()), "' (hash code ", hash_code(),
        ") but you are trying to access the resource as type '",
        port::Demangle(type_index.name()), "' (hash code ",
        type_index.hash_code(), ")");
  }
  return Status::OK();
}

std::atomic<int64_t> ResourceHandle::current_id_;

int64_t ResourceHandle::GenerateUniqueId() { return current_id_.fetch_add(1); }

string ProtoDebugString(const ResourceHandle& handle) {
  return handle.DebugString();
}

void EncodeResourceHandleList(const ResourceHandle* p, int64_t n,
                              std::unique_ptr<port::StringListEncoder> e) {
  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    p[i].AsProto(&proto);
    e->Append(proto);
  }
  e->Finalize();
}

bool DecodeResourceHandleList(std::unique_ptr<port::StringListDecoder> d,
                              ResourceHandle* ps, int64_t n) {
  std::vector<uint32> sizes(n);
  if (!d->ReadSizes(&sizes)) return false;

  ResourceHandleProto proto;
  for (int i = 0; i < n; ++i) {
    if (!proto.ParseFromArray(d->Data(sizes[i]), sizes[i])) {
      return false;
    }
    if (!ps[i].FromProto(proto).ok()) {
      return false;
    }
  }
  return true;
}

}  // namespace tensorflow
