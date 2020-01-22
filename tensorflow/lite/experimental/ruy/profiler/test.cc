/* Copyright 2020 Google LLC. All Rights Reserved.

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

#include <chrono>
#include <random>
#include <thread>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/ruy/profiler/profiler.h"
#include "tensorflow/lite/experimental/ruy/profiler/test_instrumented_library.h"
#include "tensorflow/lite/experimental/ruy/profiler/treeview.h"

namespace ruy {
namespace profiler {
namespace {

void DoSomeMergeSort(int size) {
  std::vector<int> data(size);

  std::default_random_engine engine;
  for (auto& val : data) {
    val = engine();
  }

  MergeSort(size, data.data());
}

// The purpose of this basic test is to cover the basic path that will be taken
// by a majority of users, not inspecting treeviews but just implicitly printing
// them on stdout, and to have this test enabled even when RUY_PROFILER is not
// defined, so that we have coverage for the non-RUY_PROFILER case.
TEST(ProfilerTest, MergeSortSingleThreadBasicTestEvenWithoutProfiler) {
  {
    ScopeProfile profile;
    DoSomeMergeSort(1 << 20);
  }
}

#ifdef RUY_PROFILER

TEST(ProfilerTest, MergeSortSingleThread) {
  TreeView treeview;
  {
    ScopeProfile profile;
    profile.SetUserTreeView(&treeview);
    DoSomeMergeSort(1 << 20);
  }
  Print(treeview);
  EXPECT_EQ(treeview.thread_roots().size(), 1);
  const auto& thread_root = *treeview.thread_roots().begin()->second;
  EXPECT_EQ(DepthOfTreeBelow(thread_root), 22);
  EXPECT_GE(
      WeightBelowNodeMatchingUnformatted(thread_root, "Merging sorted halves"),
      0.1 * thread_root.weight);
  EXPECT_GE(WeightBelowNodeMatchingFormatted(
                thread_root, "MergeSortRecurse (level=20, size=1)"),
            0.01 * thread_root.weight);

  TreeView treeview_collapsed;
  CollapseNodesMatchingUnformatted(treeview, 5, "MergeSort (size=%d)",
                                   &treeview_collapsed);
  Print(treeview_collapsed);
  const auto& collapsed_thread_root =
      *treeview_collapsed.thread_roots().begin()->second;
  EXPECT_EQ(DepthOfTreeBelow(collapsed_thread_root), 6);
  EXPECT_EQ(
      WeightBelowNodeMatchingUnformatted(thread_root, "MergeSort (size=%d)"),
      WeightBelowNodeMatchingUnformatted(collapsed_thread_root,
                                         "MergeSort (size=%d)"));
}

TEST(ProfilerTest, MemcpyFourThreads) {
  TreeView treeview;
  {
    ScopeProfile profile;
    profile.SetUserTreeView(&treeview);
    std::vector<std::unique_ptr<std::thread>> threads;
    for (int i = 0; i < 4; i++) {
      threads.emplace_back(new std::thread([i]() {
        ScopeLabel thread_label("worker thread #%d", i);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        ScopeLabel some_more_work_label("some more work");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }));
    }
    for (int i = 0; i < 4; i++) {
      threads[i]->join();
    }
  }
  Print(treeview);
  // Since we cleared GlobalAllThreadStacks and the current thread hasn't
  // created any ScopeLabel, only the 4 worker threads should be recorded.
  EXPECT_EQ(treeview.thread_roots().size(), 4);
  for (const auto& thread_root : treeview.thread_roots()) {
    const TreeView::Node& root_node = *thread_root.second;
    // The root node may have 1 or 2 children depending on whether there is
    // an "[other]" child.
    EXPECT_GE(root_node.children.size(), 1);
    EXPECT_LE(root_node.children.size(), 2);
    const TreeView::Node& child_node = *root_node.children[0];
    EXPECT_EQ(child_node.label.format(), "worker thread #%d");
    // There must be 2 children, since roughly half the time will be in
    // "some more work" leaving the other half in "[other]".
    EXPECT_EQ(child_node.children.size(), 2);
    const TreeView::Node& child_child_node = *child_node.children[0];
    // Since we sample every millisecond and the threads run for >= 2000
    // milliseconds, the "thread func" label should get roughly 2000 samples.
    // Not very rigorous, as we're depending on the profiler thread getting
    // scheduled, so to avoid this test being flaky, we use a much more
    // conservative value of 500, one quarter of that normal value 2000.
    EXPECT_GE(child_node.weight, 500);
    // Likewise, allow up to four times more than the normal value 2000.
    EXPECT_LE(child_node.weight, 8000);
    // Roughly half of time should be spent under the "some more work" label.
    float some_more_work_percentage =
        100.f * child_child_node.weight / child_node.weight;
    EXPECT_GE(some_more_work_percentage, 40.0f);
    EXPECT_LE(some_more_work_percentage, 60.0f);
  }
}

TEST(ProfilerTest, OneThreadAfterAnother) {
  TreeView treeview;
  {
    ScopeProfile profile;
    profile.SetUserTreeView(&treeview);
    {
      std::thread thread([]() {
        ScopeLabel thread_label("thread 0");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      });
      thread.join();
    }
    {
      std::thread thread([]() {
        ScopeLabel thread_label("thread 1");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      });
      thread.join();
    }
  }
  Print(treeview);
  EXPECT_EQ(treeview.thread_roots().size(), 2);
}

#endif  // RUY_PROFILER

}  // namespace
}  // namespace profiler
}  // namespace ruy

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
