/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt_proxy/contrib/pathways/status_annotator_util.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/user_context_registry.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/ifrt_proxy/contrib/pathways/status_annotator.pb.h"
#include "xla/tsl/platform/status_to_from_proto.h"

namespace ifrt_proxy_contrib_pathways {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::xla::ifrt::TrackedUserContextRef;

TEST(StatusAnnotatorUtilTest, SimpleAnnotateAndExpand) {
  ObjectStoreDumpProto object_store_dump;
  object_store_dump.set_device("tpu:0");

  absl::Status status = absl::InternalError("test error");
  AnnotateIfrtUserStatusWithObjectStoreDump(status, object_store_dump);

  EXPECT_THAT(xla::ifrt::ExpandUserContexts(status).message(),
              HasSubstr("tpu:0"));
}

TEST(StatusAnnotatorUtilTest, ExpandFailedDump) {
  ObjectStoreDumpProto object_store_dump;
  *object_store_dump.mutable_dump_failed() =
      tsl::StatusToProto(absl::InternalError("dump failed"));
  object_store_dump.set_device("tpu:0");

  absl::Status status = absl::InternalError("test error");
  AnnotateIfrtUserStatusWithObjectStoreDump(status, object_store_dump);

  EXPECT_THAT(xla::ifrt::ExpandUserContexts(status).message(),
              HasSubstr("tpu:0"));
  EXPECT_THAT(xla::ifrt::ExpandUserContexts(status).message(),
              HasSubstr("dump failed"));
}

static TrackedUserContextRef RegisterNewContext(
    int context, std::optional<std::string> debug_string = std::nullopt) {
  return xla::ifrt::UserContextRegistry::Get().Register(
      xla::ifrt::test_util::MakeUserContext(context, debug_string));
}

TEST(StatusAnnotatorUtilTest, DumpWithManyContentsExpandsToAllDetails) {
  ObjectStoreDumpProto object_store_dump;
  object_store_dump.set_device("tpu:0");

  std::vector<TrackedUserContextRef> user_context_refs;

  for (int context : {1000, 2000, 3000}) {
    user_context_refs.push_back(RegisterNewContext(context));

    auto* per_error_context =
        object_store_dump.mutable_dump()->add_per_error_context();
    per_error_context->set_error_context_id(context);

    for (int creator : {100, 200}) {
      auto* per_creator = per_error_context->add_per_creator();
      *per_creator->mutable_creator() = absl::StrCat("creator", creator);
      per_creator->set_ready_obj_count(context + creator + 1);
      per_creator->set_ready_total_size(context + creator + 2);
      per_creator->set_not_ready_obj_count(context + creator + 3);
      per_creator->set_not_ready_total_size(context + creator + 4);
    }
  }

  absl::Status status = absl::InternalError("test error");
  AnnotateIfrtUserStatusWithObjectStoreDump(status, object_store_dump);

  std::string expanded(xla::ifrt::ExpandUserContexts(status).message());

  for (int context : {1000, 2000, 3000}) {
    EXPECT_THAT(expanded,
                HasSubstr(absl::StrCat("TestUserContext(", context, ")")));
    for (int creator : {100, 200}) {
      EXPECT_THAT(expanded, HasSubstr(absl::StrCat("creator", creator)));
      EXPECT_THAT(expanded,
                  HasSubstr(absl::StrCat(context + creator + 1,
                                         " 'ready' buffers of total size ",
                                         context + creator + 2)));
      EXPECT_THAT(expanded,
                  HasSubstr(absl::StrCat(context + creator + 3,
                                         " 'not ready' buffers of total size ",
                                         context + creator + 4)));
    }
  }
}

TEST(StatusAnnotatorUtilTest, DumpWithDuplicateStackTraces) {
  ObjectStoreDumpProto object_store_dump;
  object_store_dump.set_device("tpu:0");

  std::vector<TrackedUserContextRef> user_context_refs;

  for (int context : {1000, 2000}) {
    user_context_refs.push_back(
        RegisterNewContext(context, /*debug_string=*/"foobar"));

    auto* per_error_context =
        object_store_dump.mutable_dump()->add_per_error_context();
    per_error_context->set_error_context_id(context);

    auto* per_creator = per_error_context->add_per_creator();
    *per_creator->mutable_creator() = "creator1";
    per_creator->set_ready_obj_count(1);
    per_creator->set_ready_total_size(1);
    per_creator->set_not_ready_obj_count(1);
    per_creator->set_not_ready_total_size(1);
  }

  absl::Status status = absl::InternalError("test error");
  AnnotateIfrtUserStatusWithObjectStoreDump(status, object_store_dump);

  std::string expanded(xla::ifrt::ExpandUserContexts(status).message());

  EXPECT_THAT(expanded, HasSubstr("foobar"));
  EXPECT_THAT(expanded, HasSubstr("2 'ready' buffers of total size 2"));
  EXPECT_THAT(expanded, HasSubstr("2 'not ready' buffers of total size 2"));

  EXPECT_EQ(expanded.find("foobar"), expanded.rfind("foobar"));
  EXPECT_EQ(expanded.find("'ready' buffers"),
            expanded.rfind("'ready' buffers"));
}

TEST(StatusAnnotatorUtilTest, CitationsWithUnknownStackTraces) {
  ObjectStoreDumpProto object_store_dump;
  object_store_dump.set_device("tpu:0");

  std::vector<TrackedUserContextRef> user_context_refs;

  for (int context : {1000, 2000, 3000}) {
    if (context != 2000) {
      user_context_refs.push_back(RegisterNewContext(context));
    }

    auto* per_error_context =
        object_store_dump.mutable_dump()->add_per_error_context();
    per_error_context->set_error_context_id(context);

    auto* per_creator = per_error_context->add_per_creator();
    *per_creator->mutable_creator() = absl::StrCat("creator", context);
    per_creator->set_ready_obj_count(context + 1);
    per_creator->set_ready_total_size(context + 2);
    per_creator->set_not_ready_obj_count(context + 3);
    per_creator->set_not_ready_total_size(context + 4);
  }

  absl::Status status = absl::InternalError("test error");
  AnnotateIfrtUserStatusWithObjectStoreDump(status, object_store_dump);

  std::string expanded(xla::ifrt::ExpandUserContexts(status).message());
  EXPECT_THAT(expanded, HasSubstr("unknown user stack"));
  EXPECT_THAT(expanded, HasSubstr("[  1]   TestUserContext(3000)"));
  EXPECT_THAT(expanded, HasSubstr("[  2]   TestUserContext(1000)"));
  EXPECT_THAT(expanded, Not(HasSubstr("[  3]")));
  EXPECT_THAT(expanded, Not(HasSubstr("[3]")));
}

}  // namespace
}  // namespace ifrt_proxy_contrib_pathways
