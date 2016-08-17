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

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class Resource : public ResourceBase {
 public:
  explicit Resource(const string& label) : label_(label) {}
  ~Resource() override {}

  string DebugString() { return strings::StrCat("R/", label_); }

 private:
  string label_;
};

class Other : public ResourceBase {
 public:
  explicit Other(const string& label) : label_(label) {}
  ~Other() override {}

  string DebugString() { return strings::StrCat("O/", label_); }

 private:
  string label_;
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
    return Status::OK();
  }));
  const string ret = r->DebugString();
  r->Unref();
  return ret;
}

static void HasError(const Status& s, const string& substr) {
  EXPECT_TRUE(StringPiece(s.ToString()).contains(substr))
      << s << ", expected substring " << substr;
}

template <typename T>
Status FindErr(const ResourceMgr& rm, const string& container,
               const string& name) {
  T* r;
  Status s = rm.Lookup(container, name, &r);
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
           "Already exists: Resource foo/bar");

  // Expected to be found.
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));
  EXPECT_EQ("R/dog", Find<Resource>(rm, "foo", "baz"));
  EXPECT_EQ("O/tiger", Find<Other>(rm, "foo", "bar"));

  // Expected to be not found.
  HasError(FindErr<Resource>(rm, "bar", "foo"), "Not found: Container bar");
  HasError(FindErr<Resource>(rm, "foo", "xxx"), "Not found: Resource foo/xxx");
  HasError(FindErr<Other>(rm, "foo", "baz"), "Not found: Resource foo/baz");

  // Delete foo/bar/Resource.
  TF_CHECK_OK(rm.Delete<Resource>("foo", "bar"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), "Not found: Resource foo/bar");

  TF_CHECK_OK(rm.Create("foo", "bar", new Resource("kitty")));
  EXPECT_EQ("R/kitty", Find<Resource>(rm, "foo", "bar"));

  // Drop the whole container foo.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), "Not found: Container foo");

  // Dropping it a second time is OK.
  TF_CHECK_OK(rm.Cleanup("foo"));
  HasError(FindErr<Resource>(rm, "foo", "bar"), "Not found: Container foo");

  // Dropping a non-existent container is also ok.
  TF_CHECK_OK(rm.Cleanup("bar"));
}

TEST(ResourceMgr, CreateOrLookup) {
  ResourceMgr rm;
  EXPECT_EQ("R/cat", LookupOrCreate<Resource>(&rm, "foo", "bar", "cat"));
  EXPECT_EQ("R/cat", LookupOrCreate<Resource>(&rm, "foo", "bar", "dog"));
  EXPECT_EQ("R/cat", Find<Resource>(rm, "foo", "bar"));

  EXPECT_EQ("O/tiger", LookupOrCreate<Other>(&rm, "foo", "bar", "tiger"));
  EXPECT_EQ("O/tiger", LookupOrCreate<Other>(&rm, "foo", "bar", "lion"));
  TF_CHECK_OK(rm.Delete<Other>("foo", "bar"));
  HasError(FindErr<Other>(rm, "foo", "bar"), "Not found: Resource foo/bar");
}

Status ComputePolicy(const string& attr_container,
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
  return Status::OK();
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
  EXPECT_EQ(Policy("", "", false), "[localhost,_0_foo,private]");
  EXPECT_EQ(Policy("", "", true), "[localhost,foo,public]");
  EXPECT_EQ(Policy("", "bar", false), "[localhost,bar,public]");
  EXPECT_EQ(Policy("", "bar", true), "[localhost,bar,public]");
  EXPECT_EQ(Policy("cat", "", false), "[cat,_1_foo,private]");
  EXPECT_EQ(Policy("cat", "", true), "[cat,foo,public]");
  EXPECT_EQ(Policy("cat", "bar", false), "[cat,bar,public]");
  EXPECT_EQ(Policy("cat", "bar", true), "[cat,bar,public]");
  EXPECT_EQ(Policy("cat.0-dog", "bar", true), "[cat.0-dog,bar,public]");
  EXPECT_EQ(Policy(".cat", "bar", true), "[.cat,bar,public]");
}

Status WrongPolicy(const string& attr_container, const string& attr_shared_name,
                   bool use_node_name_as_default) {
  string dbg;
  auto s = ComputePolicy(attr_container, attr_shared_name,
                         use_node_name_as_default, &dbg);
  CHECK(!s.ok());
  return s;
}

TEST(ContainerInfo, Error) {
  // Missing attribute.
  HasError(WrongPolicy("none", "", false), "No attr");
  HasError(WrongPolicy("", "none", false), "No attr");
  HasError(WrongPolicy("none", "none", false), "No attr");

  // Invalid container.
  HasError(WrongPolicy("12$%", "", false), "container contains invalid char");
  HasError(WrongPolicy("-cat", "", false), "container contains invalid char");

  // Invalid shared name.
  HasError(WrongPolicy("", "_foo", false), "shared_name cannot start with '_'");
}

}  // end namespace tensorflow
