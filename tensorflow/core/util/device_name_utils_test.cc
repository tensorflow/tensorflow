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

#include "tensorflow/core/util/device_name_utils.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

bool RoundTripParsedName(const string& original, const string& expected) {
  DeviceNameUtils::ParsedName p;
  if (!DeviceNameUtils::ParseFullName(original, &p)) {
    return false;
  }
  string round_tripped = DeviceNameUtils::ParsedNameToString(p);
  return (round_tripped == expected);
}

enum NamePart { kJob = 0x01, kReplica = 0x02, kTask = 0x04, kDevice = 0x08 };

bool RoundTripPartialName(int parts_to_test, const std::vector<string>& parts,
                          bool explicitDevice) {
  string original, expected;
  if (parts_to_test & kJob) {
    strings::StrAppend(&original, "/job:", parts[0]);
    strings::StrAppend(&expected, "/job:", parts[0]);
  }
  if (parts_to_test & kReplica) {
    strings::StrAppend(&original, "/replica:", parts[1]);
    strings::StrAppend(&expected, "/replica:", parts[1]);
  }
  if (parts_to_test & kTask) {
    strings::StrAppend(&original, "/task:", parts[2]);
    strings::StrAppend(&expected, "/task:", parts[2]);
  }
  if (parts_to_test & kDevice) {
    if (explicitDevice) {
      strings::StrAppend(&original, "/device:", parts[3]);
      strings::StrAppend(&expected, "/device:", parts[3]);
    } else {
      strings::StrAppend(&original, "/", parts[3]);
      strings::StrAppend(&expected,
                         "/device:", absl::AsciiStrToUpper(parts[3]));
    }
  }
  return RoundTripParsedName(original, expected);
}

}  // namespace

TEST(DeviceNameUtilsTest, Basic) {
  EXPECT_EQ(DeviceNameUtils::FullName("hello", 1, 2, "CPU", 3),
            "/job:hello/replica:1/task:2/device:CPU:3");

  {
    DeviceNameUtils::ParsedName p;
    EXPECT_FALSE(DeviceNameUtils::ParseFullName("foobar", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:123/replica:1/task:2/device:GPU:3", &p));
    EXPECT_FALSE(
        DeviceNameUtils::ParseFullName("/job:123/replica:1/task:2/gpu:", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:123/replica:1/task:2/device:gpu:", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:-1/task:2/device:GPU:3", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:1/task:-2/device:GPU:3", &p));
    EXPECT_FALSE(
        DeviceNameUtils::ParseFullName("/job:foo/replica:1/task:2/bar:3", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:1/task:2/device:GPU:3/extra", &p));
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:foo/replica:1/task:2/device:GPU:3", &p));
    EXPECT_TRUE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_TRUE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.job, "foo");
    EXPECT_EQ(p.replica, 1);
    EXPECT_EQ(p.task, 2);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 3);
  }
  {
    // Allow _ in job names.
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:foo_bar/replica:1/task:2/device:GPU:3", &p));
    EXPECT_TRUE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_TRUE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.job, "foo_bar");
    EXPECT_EQ(p.replica, 1);
    EXPECT_EQ(p.task, 2);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 3);
  }
  {
    // Allow _ in job names.
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:foo_bar/replica:1/task:2/device:GPU:3", &p));
    EXPECT_TRUE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_TRUE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.job, "foo_bar");
    EXPECT_EQ(p.replica, 1);
    EXPECT_EQ(p.task, 2);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 3);
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName("/job:*/replica:4/gpu:*", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(
        DeviceNameUtils::ParseFullName("/job:*/replica:4/device:GPU:*", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(
        DeviceNameUtils::ParseFullName("/job:*/device:GPU/replica:4", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName(
        "/job:*/replica:4/device:myspecialdevice:13", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "myspecialdevice");
    EXPECT_EQ(p.id, 13);
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName("/", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_FALSE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_FALSE(p.has_type);
    EXPECT_FALSE(p.has_id);
  }
  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(
        DeviceNameUtils::ParseFullName("/job:*/replica:4/device:GPU:5", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_TRUE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
    EXPECT_EQ(p.id, 5);
  }
  {  // Same result if we reorder the components
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseFullName("/gpu:*/job:*/replica:4", &p));
    EXPECT_FALSE(p.has_job);
    EXPECT_TRUE(p.has_replica);
    EXPECT_FALSE(p.has_task);
    EXPECT_TRUE(p.has_type);
    EXPECT_FALSE(p.has_id);
    EXPECT_EQ(p.replica, 4);
    EXPECT_EQ(p.type, "GPU");
  }

  EXPECT_TRUE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:foo/replica:1/task:2/device:GPU:4"));
  EXPECT_FALSE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:foo/replica:1/task:3/device:GPU:4"));
  EXPECT_FALSE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:foo/replica:10/task:2/device:GPU:4"));
  EXPECT_FALSE(DeviceNameUtils::IsSameAddressSpace(
      "/job:foo/replica:1/task:2/cpu:3",
      "/job:bar/replica:1/task:2/device:GPU:4"));

  EXPECT_EQ(DeviceNameUtils::LocalName("CPU", 1), "/device:CPU:1");
  EXPECT_EQ(DeviceNameUtils::LocalName("GPU", 2), "/device:GPU:2");
  EXPECT_EQ(DeviceNameUtils::LocalName("MySpecialDevice", 13),
            "/device:MySpecialDevice:13");

  EXPECT_EQ(
      DeviceNameUtils::LocalName("/job:foo/replica:1/task:2/device:CPU:3"),
      "/device:CPU:3");

  EXPECT_EQ(DeviceNameUtils::LocalName("/job:foo/replica:1/task:2/cpu:3"),
            "/device:CPU:3");

  EXPECT_EQ(
      DeviceNameUtils::LocalName("/job:foo/replica:1/task:2/device:abc:73"),
      "/device:abc:73");

  {
    DeviceNameUtils::ParsedName p;
    EXPECT_TRUE(DeviceNameUtils::ParseLocalName("CPU:10", &p));
    EXPECT_EQ(p.type, "CPU");
    EXPECT_EQ(p.id, 10);
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("cpu:abc", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("abc:", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("abc", &p));
    EXPECT_FALSE(DeviceNameUtils::ParseLocalName("myspecialdevice", &p));
  }

  // Test that all parts are round-tripped correctly.
  {
    for (int i = 0; i < 0x10; ++i) {
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "CPU:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "GPU:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "cpu:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "gpu:3"},
                                       /*explicitDevice=*/false));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "CPU:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "GPU:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "cpu:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "gpu:3"},
                                       /*explicitDevice=*/true));
      EXPECT_TRUE(RoundTripPartialName(i, {"foo", "3", "2", "someDevice:3"},
                                       /*explicitDevice=*/true));
    }
  }
}

static bool IsCSHelper(StringPiece pattern, StringPiece actual) {
  DeviceNameUtils::ParsedName p, a;
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(pattern, &p));
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(actual, &a));
  return DeviceNameUtils::IsCompleteSpecification(p, a);
}

TEST(DeviceNameUtilsTest, IsCompleteSpecification) {
  EXPECT_TRUE(IsCSHelper("/job:*", "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsCSHelper("/job:*/replica:*",
                         "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(
      IsCSHelper("/job:*/task:*", "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsCSHelper("/job:*/replica:*/task:*",
                         "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsCSHelper("/job:*/replica:*/gpu:*",
                         "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(
      IsCSHelper("/cpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(
      IsCSHelper("/device:GPU:2", "/job:worker/replica:1/task:2/device:GPU:1"));
  EXPECT_TRUE(
      IsCSHelper("/gpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));
}

static bool IsSpecHelper(StringPiece pattern, StringPiece actual) {
  DeviceNameUtils::ParsedName p, a;
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(pattern, &p));
  EXPECT_TRUE(DeviceNameUtils::ParseFullName(actual, &a));
  return DeviceNameUtils::IsSpecification(p, a);
}

TEST(DeviceNameUtilsTest, IsSpecification) {
  EXPECT_TRUE(
      IsSpecHelper("/job:*", "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/job:work/replica:1/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/job:work/replica:1"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/replica:1"));
  EXPECT_TRUE(IsSpecHelper("/job:*", "/job:work"));
  EXPECT_TRUE(IsSpecHelper("/job:*/replica:*",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:1/gpu:*",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:1/device:GPU:3",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:1/task:2",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/job:work/replica:*/task:2",
                           "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/task:*", "/job:*/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/task:2", "/job:*/replica:1/task:2/device:GPU:3"));
  EXPECT_TRUE(IsSpecHelper("/cpu:*", "/job:*/replica:1/task:2/cpu:1"));
  EXPECT_TRUE(IsSpecHelper("/cpu:0", "/cpu:0"));
  EXPECT_TRUE(
      IsSpecHelper("/gpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));

  EXPECT_FALSE(
      IsSpecHelper("/job:worker/replica:1/task:2/device:GPU:3", "/gpu:*"));
  EXPECT_FALSE(IsSpecHelper("/cpu:*", "/job:*/replica:1/task:2"));
  EXPECT_FALSE(IsSpecHelper("/cpu:*", "/job:*/replica:1/task:2/device:GPU:1"));
  EXPECT_FALSE(
      IsSpecHelper("/cpu:*", "/job:worker/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(IsSpecHelper("/device:GPU:2",
                            "/job:worker/replica:1/task:2/device:GPU:1"));
  EXPECT_FALSE(IsSpecHelper("/job:work/replica:*/task:0",
                            "/job:work/replica:1/task:2/device:GPU:3"));
  EXPECT_FALSE(IsSpecHelper("/job:work/replica:0/task:2",
                            "/job:work/replica:*/task:2/device:GPU:3"));
}

TEST(DeviceNameUtilsTest, SplitDeviceName) {
  string task;
  string device;
  EXPECT_TRUE(DeviceNameUtils::SplitDeviceName(
      "/job:foo/replica:1/task:2/cpu:1", &task, &device));
  EXPECT_EQ("/job:foo/replica:1/task:2", task);
  EXPECT_EQ("CPU:1", device);
  EXPECT_TRUE(DeviceNameUtils::SplitDeviceName(
      "/job:foo/cpu:1/task:2/replica:1", &task, &device));
  EXPECT_EQ("/job:foo/replica:1/task:2", task);
  EXPECT_EQ("CPU:1", device);
  EXPECT_TRUE(
      DeviceNameUtils::SplitDeviceName("/device:GPU:3", &task, &device));
  EXPECT_EQ("", task);
  EXPECT_EQ("GPU:3", device);
  EXPECT_FALSE(DeviceNameUtils::SplitDeviceName("gpu:3", &task, &device));
  EXPECT_FALSE(DeviceNameUtils::SplitDeviceName("/job:foo/task:2/replica:1",
                                                &task, &device));
  EXPECT_TRUE(DeviceNameUtils::SplitDeviceName("/device:myspecialdevice:3",
                                               &task, &device));
  EXPECT_EQ("", task);
  EXPECT_EQ("myspecialdevice:3", device);
}

static DeviceNameUtils::ParsedName Name(const string& str) {
  DeviceNameUtils::ParsedName ret;
  CHECK(DeviceNameUtils::ParseFullName(str, &ret)) << "Invalid name: " << str;
  return ret;
}

static void MergeDevNamesHelperImpl(const string& name_a, const string& name_b,
                                    const string& expected_merge_name,
                                    bool allow_soft_placement) {
  DeviceNameUtils::ParsedName target_a = Name(name_a);
  TF_EXPECT_OK(DeviceNameUtils::MergeDevNames(&target_a, Name(name_b),
                                              allow_soft_placement));
  DeviceNameUtils::ParsedName target_b = Name(name_b);
  TF_EXPECT_OK(DeviceNameUtils::MergeDevNames(&target_b, Name(name_a),
                                              allow_soft_placement));
  EXPECT_EQ(target_a, target_b);
  EXPECT_EQ(target_a, Name(expected_merge_name));
  EXPECT_EQ(target_b, Name(expected_merge_name));
}

static void MergeDevNamesHelper(const string& name_a, const string& name_b,
                                const string& expected_merge_name) {
  MergeDevNamesHelperImpl(name_a, name_b, expected_merge_name, false);
}

static void MergeDevNamesHelperAllowSoftPlacement(
    const string& name_a, const string& name_b,
    const string& expected_merge_name) {
  MergeDevNamesHelperImpl(name_a, name_b, expected_merge_name, true);
}

static void MergeDevNamesError(const string& name_a, const string& name_b,
                               const string& expected_error_substr) {
  DeviceNameUtils::ParsedName target_a = Name(name_a);
  Status s = DeviceNameUtils::MergeDevNames(&target_a, Name(name_b));
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(absl::StrContains(s.error_message(), expected_error_substr)) << s;
}

static void MergeOverrideHelper(const string& target, const string& name,
                                const string& expected_merge_name) {
  DeviceNameUtils::ParsedName parsed_target = Name(target);
  TF_EXPECT_OK(
      DeviceNameUtils::MergeOverrideDevNames(&parsed_target, Name(name)));
  DeviceNameUtils::ParsedName parsed_expected = Name(expected_merge_name);

  EXPECT_EQ(parsed_target, parsed_expected)
      << "parsed_target: " << DeviceNameUtils::ParsedNameToString(parsed_target)
      << " expected_name: "
      << DeviceNameUtils::ParsedNameToString(parsed_expected);
}

TEST(DeviceNameUtilsTest, MergeDevNames) {
  DeviceNameUtils::ParsedName target;

  // Idempotence tests.
  MergeDevNamesHelper("", "", "");
  MergeDevNamesHelper("/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1");

  // Merging with empty device has no effect.
  MergeDevNamesHelper("", "/job:foo", "/job:foo");
  MergeDevNamesHelper("", "/replica:2", "/replica:2");
  MergeDevNamesHelper("", "/task:7", "/task:7");
  MergeDevNamesHelper("", "/device:GPU:1", "/device:GPU:1");

  // Combining disjoint names.
  MergeDevNamesHelper("/job:foo", "/task:7", "/job:foo/task:7");
  MergeDevNamesHelper("/job:foo", "/device:GPU:1", "/job:foo/device:GPU:1");

  // Combining overlapping names.
  MergeDevNamesHelper("/job:foo/replica:0", "/replica:0/task:1",
                      "/job:foo/replica:0/task:1");

  // Wildcard tests.
  MergeDevNamesHelper("", "/gpu:*", "/gpu:*");
  MergeDevNamesHelper("/gpu:*", "/gpu:*", "/gpu:*");
  MergeDevNamesHelper("/device:GPU:1", "/gpu:*", "/device:GPU:1");

  // Incompatible components.
  MergeDevNamesError("/job:foo", "/job:bar", "incompatible jobs");
  MergeDevNamesError("/replica:0", "/replica:1", "incompatible replicas");
  MergeDevNamesError("/task:0", "/task:1", "incompatible tasks");
  MergeDevNamesError("/gpu:*", "/cpu:*", "incompatible types");
  MergeDevNamesError("/device:GPU:0", "/device:GPU:1", "incompatible ids");
}

TEST(DeviceNameUtilsTest, MergeDevNamesAllowSoftPlacement) {
  // Incompatible components with allow_soft_placement.
  MergeDevNamesHelperAllowSoftPlacement("/gpu:*", "/cpu:1", "");
  MergeDevNamesHelperAllowSoftPlacement("/cpu:*", "/device:GPU:1", "");
  MergeDevNamesHelperAllowSoftPlacement("/device:GPU:1", "/device:GPU:2",
                                        "/device:GPU:*");
}

TEST(DeviceNameUtilsTest, MergeOverrideDevNames) {
  // Idempotence tests.
  MergeOverrideHelper("", "", "");
  MergeOverrideHelper("/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1",
                      "/job:foo/replica:1/task:2/cpu:1");

  // Merging with empty device has no effect.
  MergeOverrideHelper("", "/job:foo", "/job:foo");
  MergeOverrideHelper("", "/replica:2", "/replica:2");
  MergeOverrideHelper("", "/task:7", "/task:7");
  MergeOverrideHelper("", "/device:GPU:1", "/device:GPU:1");

  // Combining disjoint names.
  MergeOverrideHelper("/job:foo", "/task:7", "/job:foo/task:7");
  MergeOverrideHelper("/job:foo", "/device:GPU:1", "/job:foo/device:GPU:1");

  // Combining overlapping names.
  MergeOverrideHelper("/job:foo/replica:0", "/replica:0/task:1",
                      "/job:foo/replica:0/task:1");

  // Wildcard tests.
  MergeOverrideHelper("", "/gpu:*", "/gpu:*");
  MergeOverrideHelper("/gpu:*", "/gpu:*", "/gpu:*");
  MergeOverrideHelper("/device:GPU:1", "/gpu:*", "/device:GPU:1");

  // Testing actual override functionality
  MergeOverrideHelper("/gpu:0", "/cpu:1", "/cpu:1");
  MergeOverrideHelper("/gpu:*", "/cpu:1", "/cpu:1");
  MergeOverrideHelper("/cpu:*", "/device:GPU:1", "/gpu:1");
  MergeOverrideHelper("/device:GPU:1", "/device:GPU:2", "/device:GPU:2");

  // Override with regular merging
  MergeOverrideHelper("/job:foo/CPU:*", "/device:GPU:1", "/job:foo/GPU:1");
  MergeOverrideHelper("/cpu:*", "/job:foo/device:GPU:1", "/job:foo/GPU:1");
  MergeOverrideHelper("/task:0/cpu:*", "/device:GPU:1", "/task:0/GPU:1");
  MergeOverrideHelper("/cpu:*", "/task:0/device:GPU:1", "/task:0/GPU:1");
}

TEST(DeviceNameUtilsTest, GetNamesForDeviceMappings) {
  DeviceNameUtils::ParsedName p =
      Name("/job:foo/replica:10/task:0/device:GPU:1");
  EXPECT_EQ(absl::StrJoin(DeviceNameUtils::GetNamesForDeviceMappings(p), ","),
            "/job:foo/replica:10/task:0/device:GPU:1,"
            "/job:foo/replica:10/task:0/gpu:1");
  p.has_task = false;
  EXPECT_EQ(absl::StrJoin(DeviceNameUtils::GetNamesForDeviceMappings(p), ","),
            "");
}

TEST(DeviceNameUtilsTest, CanonicalizeDeviceName) {
  string canonical_name;
  {
    // Good basename.
    string basename = "/job:foo/replica:10/task:0/device:CPU:0";
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/replica:10/task:0/device:CPU:1", basename, &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:1", canonical_name);
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/task:0/replica:10/device:CPU:1", basename, &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:1", canonical_name);
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/task:0/replica:10/cpu:1", basename, &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:1", canonical_name);
    TF_EXPECT_OK(DeviceNameUtils::CanonicalizeDeviceName("CPU:0", basename,
                                                         &canonical_name));
    EXPECT_EQ("/job:foo/replica:10/task:0/device:CPU:0", canonical_name);
    Status s = DeviceNameUtils::CanonicalizeDeviceName(
        "/job:foo/task:0/replica/cpu:1", basename, &canonical_name);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_EQ("", canonical_name);
  }

  {
    // Try out malformed basenames.
    string fullname = "/device:CPU:0";

    Status s = DeviceNameUtils::CanonicalizeDeviceName(
        fullname, "/device:CPU:0", &canonical_name);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_EQ("", canonical_name);
    s = DeviceNameUtils::CanonicalizeDeviceName(
        fullname, "/job:foo/task:0/replica/cpu:1", &canonical_name);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_EQ("", canonical_name);
  }
}

static void BM_ParseFullName(int iters) {
  DeviceNameUtils::ParsedName p;
  while (iters--) {
    DeviceNameUtils::ParseFullName("/job:worker/replica:3/task:0/cpu:0", &p);
  }
}
BENCHMARK(BM_ParseFullName);

}  // namespace tensorflow
