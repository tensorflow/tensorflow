#ifndef TENSORFLOW_LIB_CORE_REFCOUNT_H_
#define TENSORFLOW_LIB_CORE_REFCOUNT_H_

#include <atomic>

namespace tensorflow {
namespace core {

class RefCounted {
 public:
  // Initial reference count is one.
  RefCounted();

  // Increments reference count by one.
  void Ref() const;

  // Decrements reference count by one.  If the count remains
  // positive, returns false.  When the count reaches zero, returns
  // true and deletes this, in which case the caller must not access
  // the object afterward.
  bool Unref() const;

  // Return whether the reference count is one.
  // If the reference count is used in the conventional way, a
  // reference count of 1 implies that the current thread owns the
  // reference and no other thread shares it.
  // This call performs the test for a reference count of one, and
  // performs the memory barrier needed for the owning thread
  // to act on the object, knowing that it has exclusive access to the
  // object.
  bool RefCountIsOne() const;

 protected:
  // Make destructor protected so that RefCounted objects cannot
  // be instantiated directly. Only subclasses can be instantiated.
  virtual ~RefCounted();

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};

// Helper class to unref an object when out-of-scope.
class ScopedUnref {
 public:
  explicit ScopedUnref(RefCounted* o) : obj_(o) {}
  ~ScopedUnref() {
    if (obj_) obj_->Unref();
  }

 private:
  RefCounted* obj_;

  ScopedUnref(const ScopedUnref&) = delete;
  void operator=(const ScopedUnref&) = delete;
};

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_REFCOUNT_H_
