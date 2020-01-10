#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace core {

RefCounted::RefCounted() : ref_(1) {}

RefCounted::~RefCounted() { DCHECK_EQ(ref_.load(), 0); }

void RefCounted::Ref() const {
  DCHECK_GE(ref_.load(), 1);
  ref_.fetch_add(1, std::memory_order_relaxed);
}

bool RefCounted::Unref() const {
  DCHECK_GT(ref_.load(), 0);
  // If ref_==1, this object is owned only by the caller. Bypass a locked op
  // in that case.
  if (ref_.load(std::memory_order_acquire) == 1 || ref_.fetch_sub(1) == 1) {
    // Make DCHECK in ~RefCounted happy
    DCHECK((ref_.store(0), true));
    delete this;
    return true;
  } else {
    return false;
  }
}

bool RefCounted::RefCountIsOne() const {
  return (ref_.load(std::memory_order_acquire) == 1);
}

}  // namespace core
}  // namespace tensorflow
