/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Copied from tensorflow/core/util/ctc/ctc_beam_entry.h
// TODO(b/111524997): Remove this file.
#ifndef TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_ENTRY_H_
#define TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_ENTRY_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/kernels/ctc/ctc_loss_util.h"

namespace tflite {
namespace custom {
namespace ctc {

// The ctc_beam_search namespace holds several classes meant to be accessed only
// in case of extending the CTCBeamSearch decoder to allow custom scoring
// functions.
//
// BeamEntry is exposed through template arguments BeamScorer and BeamComparer
// of CTCBeamSearch (ctc_beam_search.h).
namespace ctc_beam_search {

struct EmptyBeamState {};

struct BeamProbability {
  BeamProbability() : total(kLogZero), blank(kLogZero), label(kLogZero) {}
  void Reset() {
    total = kLogZero;
    blank = kLogZero;
    label = kLogZero;
  }
  float total;
  float blank;
  float label;
};

template <class CTCBeamState>
class BeamRoot;

template <class CTCBeamState = EmptyBeamState>
struct BeamEntry {
  // BeamRoot<CTCBeamState>::AddEntry() serves as the factory method.
  friend BeamEntry<CTCBeamState>* BeamRoot<CTCBeamState>::AddEntry(
      BeamEntry<CTCBeamState>* p, int l);
  inline bool Active() const { return newp.total != kLogZero; }
  // Return the child at the given index, or construct a new one in-place if
  // none was found.
  BeamEntry& GetChild(int ind) {
    auto entry = children.emplace(ind, nullptr);
    auto& child_entry = entry.first->second;
    // If this is a new child, populate the BeamEntry<CTCBeamState>*.
    if (entry.second) {
      child_entry = beam_root->AddEntry(this, ind);
    }
    return *child_entry;
  }
  std::vector<int> LabelSeq(bool merge_repeated) const {
    std::vector<int> labels;
    int prev_label = -1;
    const BeamEntry* c = this;
    while (c->parent != nullptr) {  // Checking c->parent to skip root leaf.
      if (!merge_repeated || c->label != prev_label) {
        labels.push_back(c->label);
      }
      prev_label = c->label;
      c = c->parent;
    }
    std::reverse(labels.begin(), labels.end());
    return labels;
  }

  BeamEntry<CTCBeamState>* parent;
  int label;
  // All instances of child BeamEntry are owned by *beam_root.
  std::unordered_map<int, BeamEntry<CTCBeamState>*> children;
  BeamProbability oldp;
  BeamProbability newp;
  CTCBeamState state;

 private:
  // Constructor giving parent, label, and the beam_root.
  // The object pointed to by p cannot be copied and should not be moved,
  // otherwise parent will become invalid.
  // This private constructor is only called through the factory method
  // BeamRoot<CTCBeamState>::AddEntry().
  BeamEntry(BeamEntry* p, int l, BeamRoot<CTCBeamState>* beam_root)
      : parent(p), label(l), beam_root(beam_root) {}
  BeamRoot<CTCBeamState>* beam_root;

  BeamEntry(const BeamEntry&) = delete;
  void operator=(const BeamEntry&) = delete;
};

// This class owns all instances of BeamEntry.  This is used to avoid recursive
// destructor call during destruction.
template <class CTCBeamState = EmptyBeamState>
class BeamRoot {
 public:
  BeamRoot(BeamEntry<CTCBeamState>* p, int l) { root_entry_ = AddEntry(p, l); }
  BeamRoot(const BeamRoot&) = delete;
  BeamRoot& operator=(const BeamRoot&) = delete;

  BeamEntry<CTCBeamState>* AddEntry(BeamEntry<CTCBeamState>* p, int l) {
    auto* new_entry = new BeamEntry<CTCBeamState>(p, l, this);
    beam_entries_.emplace_back(new_entry);
    return new_entry;
  }
  BeamEntry<CTCBeamState>* RootEntry() const { return root_entry_; }

 private:
  BeamEntry<CTCBeamState>* root_entry_ = nullptr;
  std::vector<std::unique_ptr<BeamEntry<CTCBeamState>>> beam_entries_;
};

// BeamComparer is the default beam comparer provided in CTCBeamSearch.
template <class CTCBeamState = EmptyBeamState>
class BeamComparer {
 public:
  virtual ~BeamComparer() {}
  virtual bool inline operator()(const BeamEntry<CTCBeamState>* a,
                                 const BeamEntry<CTCBeamState>* b) const {
    return a->newp.total > b->newp.total;
  }
};

}  // namespace ctc_beam_search

}  // namespace ctc
}  // namespace custom
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CTC_CTC_BEAM_ENTRY_H_
