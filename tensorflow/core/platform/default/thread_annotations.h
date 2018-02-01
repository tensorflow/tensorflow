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

// This header file contains the macro definitions for thread safety
// annotations that allow the developers to document the locking policies
// of their multi-threaded code. The annotations can also help program
// analysis tools to identify potential thread safety issues.
//
// The primary documentation on these annotations is external:
// http://clang.llvm.org/docs/ThreadSafetyAnalysis.html
//
// The annotations are implemented using compiler attributes.
// Using the macros defined here instead of the raw attributes allows
// for portability and future compatibility.
//
// When referring to mutexes in the arguments of the attributes, you should
// use variable names or more complex expressions (e.g. my_object->mutex_)
// that evaluate to a concrete mutex object whenever possible. If the mutex
// you want to refer to is not in scope, you may use a member pointer
// (e.g. &MyClass::mutex_) to refer to a mutex in some (unknown) object.
//

#ifndef TENSORFLOW_PLATFORM_DEFAULT_THREAD_ANNOTATIONS_H_
#define TENSORFLOW_PLATFORM_DEFAULT_THREAD_ANNOTATIONS_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/thread_annotations.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/thread_annotations.h

#if defined(__clang__) && (!defined(SWIG))
#define THREAD_ANNOTATION_ATTRIBUTE__(x) __attribute__((x))
#else
#define THREAD_ANNOTATION_ATTRIBUTE__(x)  // no-op
#endif

// Document if a shared variable/field needs to be protected by a mutex.
// GUARDED_BY allows the user to specify a particular mutex that should be
// held when accessing the annotated variable.  GUARDED_VAR indicates that
// a shared variable is guarded by some unspecified mutex, for use in rare
// cases where a valid mutex expression cannot be specified.
#define GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))
#define GUARDED_VAR  // no-op

// Document if the memory location pointed to by a pointer should be guarded
// by a mutex when dereferencing the pointer.  PT_GUARDED_VAR is analogous to
// GUARDED_VAR.   Note that a pointer variable to a shared memory location
// could itself be a shared variable. For example, if a shared global pointer
// q, which is guarded by mu1, points to a shared memory location that is
// guarded by mu2, q should be annotated as follows:
//     int *q GUARDED_BY(mu1) PT_GUARDED_BY(mu2);
#define PT_GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_by(x))
#define PT_GUARDED_VAR  // no-op

// Document the acquisition order between locks that can be held
// simultaneously by a thread. For any two locks that need to be annotated
// to establish an acquisition order, only one of them needs the annotation.
// (i.e. You don't have to annotate both locks with both ACQUIRED_AFTER
// and ACQUIRED_BEFORE.)
#define ACQUIRED_AFTER(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquired_after(__VA_ARGS__))

#define ACQUIRED_BEFORE(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquired_before(__VA_ARGS__))

#define ACQUIRE(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquire_capability(__VA_ARGS__))

#define ACQUIRE_SHARED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

#define RELEASE(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(release_capability(__VA_ARGS__))

// Document a function that expects a mutex to be held prior to entry.
// The mutex is expected to be held both on entry to and exit from the
// function.
#define EXCLUSIVE_LOCKS_REQUIRED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(exclusive_locks_required(__VA_ARGS__))

#define SHARED_LOCKS_REQUIRED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(shared_locks_required(__VA_ARGS__))

// Document the locks acquired in the body of the function. These locks
// cannot be held when calling this function (for instance, when the
// mutex implementation is non-reentrant).
#define LOCKS_EXCLUDED(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

// Document a function that returns a mutex without acquiring it.  For example,
// a public getter method that returns a pointer to a private mutex should
// be annotated with LOCK_RETURNED.
#define LOCK_RETURNED(x) THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

// Document if a class/type is a lockable type (such as the Mutex class).
#define LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(lockable)

// Document if a class does RAII locking (such as the MutexLock class).
// The constructor should use LOCK_FUNCTION to specify the mutex that is
// acquired, and the destructor should use UNLOCK_FUNCTION with no arguments;
// the analysis will assume that the destructor unlocks whatever the
// constructor locked.
#define SCOPED_LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

// Document functions that acquire a lock in the body of a function, and do
// not release it.
#define EXCLUSIVE_LOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(exclusive_lock_function(__VA_ARGS__))

#define SHARED_LOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(shared_lock_function(__VA_ARGS__))

// Document functions that expect a lock to be held on entry to the function,
// and release it in the body of the function.
#define UNLOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(unlock_function(__VA_ARGS__))

// Document functions that try to acquire a lock, and return success or failure
// (or a non-boolean value that can be interpreted as a boolean).
// The first argument should be true for functions that return true on success,
// or false for functions that return false on success. The second argument
// specifies the mutex that is locked on success. If unspecified, it is assumed
// to be 'this'.
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(exclusive_trylock_function(__VA_ARGS__))

#define SHARED_TRYLOCK_FUNCTION(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(shared_trylock_function(__VA_ARGS__))

// Document functions that dynamically check to see if a lock is held, and fail
// if it is not held.
#define ASSERT_EXCLUSIVE_LOCK(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(assert_exclusive_lock(__VA_ARGS__))

#define ASSERT_SHARED_LOCK(...) \
  THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_lock(__VA_ARGS__))

// Turns off thread safety checking within the body of a particular function.
// This is used as an escape hatch for cases where either (a) the function
// is correct, but the locking is more complicated than the analyzer can handle,
// or (b) the function contains race conditions that are known to be benign.
#define NO_THREAD_SAFETY_ANALYSIS \
  THREAD_ANNOTATION_ATTRIBUTE__(no_thread_safety_analysis)

// TS_UNCHECKED should be placed around lock expressions that are not valid
// C++ syntax, but which are present for documentation purposes.  These
// annotations will be ignored by the analysis.
#define TS_UNCHECKED(x) ""

namespace tensorflow {
namespace thread_safety_analysis {

// Takes a reference to a guarded data member, and returns an unguarded
// reference.
template <class T>
inline const T& ts_unchecked_read(const T& v) NO_THREAD_SAFETY_ANALYSIS {
  return v;
}

template <class T>
inline T& ts_unchecked_read(T& v) NO_THREAD_SAFETY_ANALYSIS {
  return v;
}
}  // namespace thread_safety_analysis
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DEFAULT_THREAD_ANNOTATIONS_H_
