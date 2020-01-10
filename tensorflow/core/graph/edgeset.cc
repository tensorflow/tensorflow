#include "tensorflow/core/graph/edgeset.h"

namespace tensorflow {

std::pair<EdgeSet::const_iterator, bool> EdgeSet::insert(value_type value) {
  RegisterMutation();
  const_iterator ci;
  ci.Init(this);
  auto s = get_set();
  if (!s) {
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == value) {
        ci.array_iter_ = &ptrs_[i];
        return std::make_pair(ci, false);
      }
    }
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == nullptr) {
        ptrs_[i] = value;
        ci.array_iter_ = &ptrs_[i];
        return std::make_pair(ci, true);
      }
    }
    // array is full. convert to set.
    s = new std::set<const Edge*>;
    for (int i = 0; i < kInline; i++) {
      s->insert(static_cast<const Edge*>(ptrs_[i]));
    }
    ptrs_[0] = this;
    ptrs_[1] = s;
    // fall through.
  }
  auto p = s->insert(value);
  ci.tree_iter_ = p.first;
  return std::make_pair(ci, p.second);
}

EdgeSet::size_type EdgeSet::erase(key_type key) {
  RegisterMutation();
  auto s = get_set();
  if (!s) {
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i] == key) {
        size_t n = size();
        ptrs_[i] = ptrs_[n - 1];
        ptrs_[n - 1] = nullptr;
        return 1;
      }
    }
    return 0;
  } else {
    return s->erase(key);
  }
}

}  // namespace tensorflow
