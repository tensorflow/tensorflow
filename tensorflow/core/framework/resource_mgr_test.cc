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

#include "tensorflow/core/framework/resource_mgr.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

class Resource : public ResourceBase {
 public:
  explicit Resource(const string& label) : label_(label) {}
  ~Resource() override {}

  string DebugString() const override { return absl::StrCat("R/", label_); }

 private:
  string label_;
};

class Other : public ResourceBase {
 public:
  explicit Other(const string& label) : label_(label) {}
  ~Other() override {}

  string DebugString() const override { return absl::StrCat("O/", label_); }

 private:
  string label_;
};

class Finalizable : public ResourceBase {
 public:
  explicit Finalizable(int* absl_nonnull finalize_count)
      : finalize_count_(*finalize_count) {}
  ~Finalizable() override = default;

  std::string DebugString() const override { return "Finalizable"; }
  void Finalize() override { ++finalize_count_; }

 private:
  int& finalize_count_;
};

template <typename T>
string Find(const ResourceMgr& rm, const string& container,
            const string& name) {
  T* r;
  TF_CHECK_OK(rm.Lookup(container, name, &r));
  const string ret = r->DebugString();
  r->Unref();
  return ret;
}

template <typename T>
string LookupOrCreate(ResourceMgr* rm, const string& container,
                      const string& name, const string& label) {
  T* r;
  TF_CHECK_OK(rm->LookupOrCreate<T>(container, name, &r, [&label](T** ret) {
    *ret = new T(label);
    return absl::OkStatus();
  }));
  const string ret = r->DebugString();
  r->Unref();
  return ret;
}

static void HasError(const absl::Status& s, const error::Code code,
                     const string& substr) {
  EXPECT_EQ(s.code(), code);
  EXPECT_TRUE(absl::StrContains(s.message(), substr))
      << s << ", expected substring " << substr;
}

template <typename T>
absl::Status FindErr(const ResourceMgr& rm, const string& container,
                     const string& name) {
  T* r;
  absl::Status s = rm.Lookup(container, name, &r);
  CHECK(!s.ok());
  return s;
}

TEST(ResourceMgrTest, Basic) {
  ResourceMgr rm;
  TF_CHECK_OK(rm.Create("foo", "bar", new Resource("cat")));
  TF_CHECK_OK(rm.Create("foo", "baz", new Resource("dog")));
  TF_CHECK_OK(rm.Create("foo", "bar", new Other("tiger")));

  // Expected to fail.
  HasError(rm.Create("foo", "bar", new Resource("kitty")),
           error::ALREADY_EXISTS, "Resource foo/bar");

  // Expected to be found.
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));
  EXPECT_EQ("R/dog", Find<Resource>(rm, "foo", "baz"));
  EXPECT_EQ("O/tiger", Find<Other>(rm, "foo", "bar"));

  // Expected to be not found.
  HasError(FindErr<Resource>(rm, "bar", "foo"), error::NOT_FOUND,
           "Container bar");
  HasError(FindErr<Resource>(rm, "foo", "xxx"), error::NOT_FOUND,
           "Resource foo/xxx");
  HasError(FindErr<Other>(rm, "foo", "baz"), error::NOT_FOUND,
           "Resource foo/baz");

  // Delete foo/bar/Resource.
  TF_CHECK_OK(rm.Delete<Resource>("foo", "bar"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Resource foo/bar");
  // Deleting foo/bar/Resource a second time is not OK.
  HasError(rm.Delete<Resource>("foo", "bar"), error::NOT_FOUND,
           "Resource foo/bar");

  TF_CHECK_OK(rm.Create("foo", "bar", new Resource("kitty")));
  EXPECT_EQ("R/kitty", Find<Resource>(rm, "foo", "bar"));

  // Drop the whole container foo.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping it a second time is OK.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping a non-existent container is also ok.
  TF_CHECK_OK(rm.Cleanup("bar"));
}

TEST(ResourceMgrTest, CreateUnowned) {
  core::RefCountPtr<Resource> cat{new Resource("cat")};
  core::RefCountPtr<Resource> kitty{new Resource("kitty")};

  ASSERT_TRUE(cat->RefCountIsOne());
  ASSERT_TRUE(kitty->RefCountIsOne());

  ResourceMgr rm;

  TF_CHECK_OK(rm.CreateUnowned("foo", "bar", cat.get()));
  EXPECT_TRUE(cat->RefCountIsOne());

  // Expected to fail.
  HasError(rm.CreateUnowned("foo", "bar", kitty.get()), error::ALREADY_EXISTS,
           "Resource foo/bar");
  EXPECT_TRUE(kitty->RefCountIsOne());

  // Expected to be found.
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));

  // Expected to be not found.
  HasError(FindErr<Resource>(rm, "bar", "foo"), error::NOT_FOUND,
           "Container bar");
  HasError(FindErr<Resource>(rm, "foo", "xxx"), error::NOT_FOUND,
           "Resource foo/xxx");

  // Deleting foo/bar/Resource is not OK because it is not owned by the manager.
  HasError(rm.Delete<Resource>("foo", "bar"), error::INTERNAL,
           "Cannot delete an unowned Resource foo/bar");

  TF_CHECK_OK(rm.CreateUnowned("foo", "bar", kitty.get()));
  EXPECT_TRUE(kitty->RefCountIsOne());
  EXPECT_EQ("R/kitty", Find<Resource>(rm, "foo", "bar"));

  {
    core::RefCountPtr<Resource> dog{new Resource("dog")};
    TF_CHECK_OK(rm.CreateUnowned("foo", "bark", dog.get()));
    EXPECT_EQ("R/dog", Find<Resource>(rm, "foo", "bark"));
    EXPECT_EQ(1, dog->WeakRefCount());
    {
      ResourceMgr rm1;
      TF_CHECK_OK(rm1.CreateUnowned("foo", "bark", dog.get()));
      EXPECT_EQ("R/dog", Find<Resource>(rm1, "foo", "bark"));
      EXPECT_EQ(2, dog->WeakRefCount());
    }
    // If manager goes out of scope, the resource loses the weak ref.
    EXPECT_EQ(1, dog->WeakRefCount());
  }
  // If resource goes out of scope, the look up reports not found.
  HasError(FindErr<Resource>(rm, "foo", "bark"), error::NOT_FOUND,
           "Resource foo/bark");

  // Drop the whole container foo.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping it a second time is OK.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), error::NOT_FOUND,
           "Container foo");

  // Dropping a non-existent container is also ok.
  TF_CHECK_OK(rm.Cleanup("bar"));

  EXPECT_TRUE(cat->RefCountIsOne());
  EXPECT_TRUE(kitty->RefCountIsOne());
}

TEST(ResourceMgrTest, CreateOrLookup) {
  ResourceMgr rm;
  EXPECT_EQ("R/cat", LookupOrCreate<Resource>(&rm, "foo", "bar", "cat"));
  EXPECT_EQ("R/cat", LookupOrCreate<Resource>(&rm, "foo", "bar", "dog"));
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));

  EXPECT_EQ("O/tiger", LookupOrCreate<Other>(&rm, "foo", "bar", "tiger"));
  EXPECT_EQ("O/tiger", LookupOrCreate<Other>(&rm, "foo", "bar", "lion"));
  TF_CHECK_OK(rm.Delete<Other>("foo", "bar"));
  HasError(FindErr<Other>(rm, "foo", "bar"), error::NOT_FOUND,
           "Resource foo/bar");
}

TEST(ResourceMgrTest, CreateOrLookupRaceCondition) {
  ResourceMgr rm;
  std::atomic<int> atomic_int(0);
  {
    thread::ThreadPool threads(Env::Default(), "racing_creates", 2);
    for (int i = 0; i < 2; i++) {
      threads.Schedule([&rm, &atomic_int] {
        Resource* r;
        TF_CHECK_OK(rm.LookupOrCreate<Resource>(
            "container", "resource-name", &r, [&atomic_int](Resource** ret) {
              // Maximize chance of encountering race condition if one exists.
              Env::Default()->SleepForMicroseconds(1 * 1000 * 1000);
              atomic_int += 1;
              *ret = new Resource("label");
              return absl::OkStatus();
            }));
        r->Unref();
      });
    }
  }
  // Resource creator function should always run exactly once.
  EXPECT_EQ(1, atomic_int);
}

TEST(ResourceMgrTest, Finalize) {
  ResourceMgr rm;
  int finalize_count_ = 0;
  TF_ASSERT_OK(rm.Create("container", "resource-name",
                         new Finalizable(&finalize_count_)));
  EXPECT_EQ(finalize_count_, 0);

  // Finalizable::Finalize called.
  rm.Finalize();
  EXPECT_EQ(finalize_count_, 1);
}

TEST(ResourceMgrTest, MultipleFinalize) {
  ResourceMgr rm;
  int finalize_count_ = 0;
  TF_ASSERT_OK(rm.Create("container", "resource-name",
                         new Finalizable(&finalize_count_)));
  EXPECT_EQ(finalize_count_, 0);

  // Finalizable::Finalize should be called only once.
  rm.Finalize();
  EXPECT_EQ(finalize_count_, 1);
  rm.Finalize();
  EXPECT_EQ(finalize_count_, 1);
}

TEST(ResourceMgrTest, CreateFailAfterFinalize) {
  ResourceMgr rm;
  rm.Finalize();

  // Create should fail after finalization.
  int finalize_count_ = 0;
  Finalizable* finalizable = new Finalizable(&finalize_count_);
  EXPECT_THAT(rm.Create("container", "resource-name", finalizable),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     HasSubstr("ResourceMgr is finalized")));
  finalizable->Unref();
}

TEST(ResourceMgrTest, CreateUnownedFailAfterFinalize) {
  ResourceMgr rm;
  rm.Finalize();

  // Create should fail after finalization.
  int finalize_count_ = 0;
  Finalizable* finalizable = new Finalizable(&finalize_count_);
  EXPECT_THAT(rm.CreateUnowned("container", "resource-name", finalizable),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     HasSubstr("ResourceMgr is finalized")));
  finalizable->Unref();
}

absl::Status ComputePolicy(const string& attr_container,
                           const string& attr_shared_name,
                           bool use_node_name_as_default, string* result) {
  ContainerInfo cinfo;
  ResourceMgr rmgr;
  NodeDef ndef;
  ndef.set_name("foo");
  if (attr_container != "none") {
    AddNodeAttr("container", attr_container, &ndef);
  }
  if (attr_shared_name != "none") {
    AddNodeAttr("shared_name", attr_shared_name, &ndef);
  }
  TF_RETURN_IF_ERROR(cinfo.Init(&rmgr, ndef, use_node_name_as_default));
  *result = cinfo.DebugString();
  return absl::OkStatus();
}

string Policy(const string& attr_container, const string& attr_shared_name,
              bool use_node_name_as_default) {
  string ret;
  TF_CHECK_OK(ComputePolicy(attr_container, attr_shared_name,
                            use_node_name_as_default, &ret));
  return ret;
}

TEST(ContainerInfo, Basic) {
  // Correct cases.
  EXPECT_TRUE(RE2::FullMatch(Policy("", "", false),
                             "\\[localhost,_\\d+_foo,private\\]"));
  EXPECT_EQ(Policy("", "", true), "[localhost,foo,public]");
  EXPECT_EQ(Policy("", "bar", false), "[localhost,bar,public]");
  EXPECT_EQ(Policy("", "bar", true), "[localhost,bar,public]");
  EXPECT_TRUE(
      RE2::FullMatch(Policy("cat", "", false), "\\[cat,_\\d+_foo,private\\]"));
  EXPECT_EQ(Policy("cat", "", true), "[cat,foo,public]");
  EXPECT_EQ(Policy("cat", "bar", false), "[cat,bar,public]");
  EXPECT_EQ(Policy("cat", "bar", true), "[cat,bar,public]");
  EXPECT_EQ(Policy("cat.0-dog", "bar", true), "[cat.0-dog,bar,public]");
  EXPECT_EQ(Policy(".cat", "bar", true), "[.cat,bar,public]");
}

absl::Status WrongPolicy(const string& attr_container,
                         const string& attr_shared_name,
                         bool use_node_name_as_default) {
  string dbg;
  auto s = ComputePolicy(attr_container, attr_shared_name,
                         use_node_name_as_default, &dbg);
  CHECK(!s.ok());
  return s;
}

TEST(ContainerInfo, Error) {
  // Missing attribute.
  HasError(WrongPolicy("none", "", false), error::NOT_FOUND, "No attr");
  HasError(WrongPolicy("", "none", false), error::NOT_FOUND, "No attr");
  HasError(WrongPolicy("none", "none", false), error::NOT_FOUND, "No attr");

  // Invalid container.
  HasError(WrongPolicy("12$%", "", false), error::INVALID_ARGUMENT,
           "container contains invalid char");
  HasError(WrongPolicy("-cat", "", false), error::INVALID_ARGUMENT,
           "container contains invalid char");

  // Invalid shared name.
  HasError(WrongPolicy("", "_foo", false), error::INVALID_ARGUMENT,
           "shared_name cannot start with '_'");
}

// Stub DeviceBase subclass which only sets a device name, for testing resource
// handles.
class StubDevice : public DeviceBase {
 public:
  explicit StubDevice(const string& name) : DeviceBase(nullptr) {
    attr_.set_name(name);
  }

  Allocator* GetAllocator(AllocatorAttributes) override {
    return cpu_allocator();
  }

  const DeviceAttributes& attributes() const override { return attr_; }
  const string& name() const override { return attr_.name(); }

 private:
  DeviceAttributes attr_;
};

// Empty stub resource for testing resource handles.
class StubResource : public ResourceBase {
 public:
  string DebugString() const override { return ""; }
  int value_{0};
};

TEST(ResourceHandleTest, CRUD) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  {
    auto* r = new StubResource();
    r->value_ = 42;
    TF_EXPECT_OK(CreateResource(&ctx, p, r));
  }
  {
    core::RefCountPtr<StubResource> r;
    TF_ASSERT_OK(LookupResource(&ctx, p, &r));
    ASSERT_TRUE(r != nullptr);
    EXPECT_EQ(r->value_, 42);
  }
  {
    TF_EXPECT_OK(DeleteResource<StubResource>(&ctx, p));
    core::RefCountPtr<StubResource> unused;
    EXPECT_FALSE(LookupResource(&ctx, p, &unused).ok());
  }
}

TEST(ResourceHandleTest, ResourceFromValidIntInput) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 1);

  ResourceHandleProto proto;
  proto.set_device("cpu:0");
  proto.set_container("test_container");
  proto.set_name("test_var");
  auto handle = std::make_unique<ResourceHandle>(proto);
  auto expected_summary =
      "ResourceHandle(name=\"test_var\", device=\"cpu:0\", "
      "container=\"test_container\", type=\"\", dtype and shapes : \"[  ]\")";
  EXPECT_EQ(handle->SummarizeValue(), expected_summary);

  Tensor arg0(DT_RESOURCE, TensorShape({2}));
  arg0.flat<ResourceHandle>()(0) = *handle;
  std::vector<tensorflow::TensorValue> inputs{TensorValue(new Tensor(arg0))};
  params.inputs = inputs;

  ResourceHandle get_int_handle;
  TF_ASSERT_OK(HandleFromInput(&ctx, 0, &get_int_handle));
  EXPECT_EQ(get_int_handle.SummarizeValue(), expected_summary);
  delete inputs.at(0).tensor;
}

TEST(ResourceHandleTest, ResourceFromInvalidIntInput) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle get_int_handle;
  EXPECT_FALSE(HandleFromInput(&ctx, 0, &get_int_handle).ok());
}

TEST(ResourceHandleTest, ResourceFromIntInputWithoutResource) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 1);

  std::vector<tensorflow::TensorValue> inputs{TensorValue(new Tensor())};
  params.inputs = inputs;

  ResourceHandle get_int_handle;
  EXPECT_FALSE(HandleFromInput(&ctx, 0, &get_int_handle).ok());
  delete inputs.at(0).tensor;
}

TEST(ResourceHandleTest, LookupDeleteGenericResource) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  {
    auto* r = new StubResource();
    r->value_ = 42;
    TF_EXPECT_OK(CreateResource(&ctx, p, r));
  }
  {
    ResourceBase* r;
    TF_ASSERT_OK(LookupResource(&ctx, p, &r));
    ASSERT_TRUE(r != nullptr);
    core::ScopedUnref unref(r);
    EXPECT_EQ(static_cast<StubResource*>(r)->value_, 42);
  }
  {
    TF_EXPECT_OK(DeleteResource(&ctx, p));
    ResourceBase* unused;
    EXPECT_FALSE(LookupResource(&ctx, p, &unused).ok());
  }
}

TEST(ResourceHandleTest, DifferentDevice) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  ResourceMgr other_resource_mgr("");
  OpKernelContext::Params other_params;
  other_params.resource_manager = &other_resource_mgr;
  StubDevice other_device("other_device_name");
  other_params.device = &other_device;
  OpKernelContext other_ctx(&other_params, 0);

  auto* r = new StubResource();
  ASSERT_FALSE(CreateResource(&other_ctx, p, r).ok());
  r->Unref();
}

// Other stub resource to test type-checking of resource handles.
class OtherStubResource : public ResourceBase {
 public:
  string DebugString() const override { return ""; }
};

TEST(ResourceHandleTest, DifferentType) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  auto* r = new OtherStubResource;
  ASSERT_FALSE(CreateResource(&ctx, p, r).ok());
  r->Unref();
}

TEST(ResourceHandleTest, DeleteUsingResourceHandle) {
  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  StubDevice device("device_name");
  params.device = &device;
  OpKernelContext ctx(&params, 0);

  ResourceHandle p =
      MakeResourceHandle<StubResource>(&ctx, "container", "name");

  StubResource* r = new StubResource;
  TF_EXPECT_OK(CreateResource(&ctx, p, r));

  core::RefCountPtr<StubResource> lookup_r;
  TF_EXPECT_OK(LookupResource<StubResource>(&ctx, p, &lookup_r));
  EXPECT_EQ(lookup_r.get(), r);

  TF_EXPECT_OK(DeleteResource(&ctx, p));
  EXPECT_NE(LookupResource<StubResource>(&ctx, p, &lookup_r).ok(), true);
}

}  // end namespace tensorflow
