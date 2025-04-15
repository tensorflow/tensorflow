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

#ifndef TENSORFLOW_TSL_PLATFORM_THREAD_ANNOTATIONS_H_
#define TENSORFLOW_TSL_PLATFORM_THREAD_ANNOTATIONS_H_

// IWYU pragma: private, include "tsl/platform/thread_annotations.h"
// IWYU pragma: friend third_party/tensorflow/tsl/platform/thread_annotations.h

#if defined(__clang__) && (!defined(SWIG))
#define TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(x) __attribute__((x))
#else
#define TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(x)  // no-op
#endif

// Document if a shared variable/field needs to be protected by a mutex.
// TF_GUARDED_BY allows the user to specify a particular mutex that should be
// held when accessing the annotated variable.  GUARDED_VAR indicates that
// a shared variable is guarded by some unspecified mutex, for use in rare
// cases where a valid mutex expression cannot be specified.
#define TF_GUARDED_BY(x) TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(guarded_by(x))
#define GUARDED_VAR  // no-op

// Document if the memory location pointed to by a pointer should be guarded
// by a mutex when dereferencing the pointer.  PT_GUARDED_VAR is analogous to
// GUARDED_VAR.   Note that a pointer variable to a shared memory location
// could itself be a shared variable. For example, if a shared global pointer
// q, which is guarded by mu1, points to a shared memory location that is
// guarded by mu2, q should be annotated as follows:
//     int *q TF_GUARDED_BY(mu1) TF_PT_GUARDED_BY(mu2);
#define TF_PT_GUARDED_BY(x) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(pt_guarded_by(x))
#define TF_PT_GUARDED_VAR  // no-op

// Document the acquisition order between locks that can be held
// simultaneously by a thread. For any two locks that need to be annotated
// to establish an acquisition order, only one of them needs the annotation.
// (i.e. You don't have to annotate both locks with both TF_ACQUIRED_AFTER
// and TF_ACQUIRED_BEFORE.)
#define TF_ACQUIRED_AFTER(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(acquired_after(__VA_ARGS__))

#define TF_ACQUIRED_BEFORE(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(acquired_before(__VA_ARGS__))

#define TF_ACQUIRE(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(__VA_ARGS__))

#define TF_ACQUIRE_SHARED(...)             \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE( \
      acquire_shared_capability(__VA_ARGS__))

#define TF_RELEASE(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(release_capability(__VA_ARGS__))

// Document a function that expects a mutex to be held prior to entry.
// The mutex is expected to be held both on entry to and exit from the
// function.
#define TF_EXCLUSIVE_LOCKS_REQUIRED(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(exclusive_locks_required(__VA_ARGS__))

#define TF_SHARED_LOCKS_REQUIRED(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(shared_locks_required(__VA_ARGS__))

// Document the locks acquired in the body of the function. These locks
// cannot be held when calling this function (for instance, when the
// mutex implementation is non-reentrant).
#define TF_LOCKS_EXCLUDED(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(locks_excluded(__VA_ARGS__))

// Document a function that returns a mutex without acquiring it.  For example,
// a public getter method that returns a pointer to a private mutex should
// be annotated with TF_LOCK_RETURNED.
#define TF_LOCK_RETURNED(x) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(lock_returned(x))

// Document if a class/type is a lockable type (such as the Mutex class).
#define TF_LOCKABLE TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(lockable)

// Document if a class does RAII locking (such as the MutexLock class).
// The constructor should use LOCK_FUNCTION to specify the mutex that is
// acquired, and the destructor should use TF_UNLOCK_FUNCTION with no arguments;
// the analysis will assume that the destructor unlocks whatever the
// constructor locked.
#define TF_SCOPED_LOCKABLE \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(scoped_lockable)

// Document functions that acquire a lock in the body of a function, and do
// not release it.
#define TF_EXCLUSIVE_LOCK_FUNCTION(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(exclusive_lock_function(__VA_ARGS__))

#define TF_SHARED_LOCK_FUNCTION(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(shared_lock_function(__VA_ARGS__))

// Document functions that expect a lock to be held on entry to the function,
// and release it in the body of the function.
#define TF_UNLOCK_FUNCTION(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(unlock_function(__VA_ARGS__))

// Document functions that try to acquire a lock, and return success or failure
// (or a non-boolean value that can be interpreted as a boolean).
// The first argument should be true for functions that return true on success,
// or false for functions that return false on success. The second argument
// specifies the mutex that is locked on success. If unspecified, it is assumed
// to be 'this'.
#define TF_EXCLUSIVE_TRYLOCK_FUNCTION(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE( \
      exclusive_trylock_function(__VA_ARGS__))

#define TF_SHARED_TRYLOCK_FUNCTION(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(shared_trylock_function(__VA_ARGS__))

// Document functions that dynamically check to see if a lock is held, and fail
// if it is not held.
#define TF_ASSERT_EXCLUSIVE_LOCK(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(assert_exclusive_lock(__VA_ARGS__))

#define TF_ASSERT_SHARED_LOCK(...) \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(assert_shared_lock(__VA_ARGS__))

// Turns off thread safety checking within the body of a particular function.
// This is used as an escape hatch for cases where either (a) the function
// is correct, but the locking is more complicated than the analyzer can handle,
// or (b) the function contains race conditions that are known to be benign.
#define TF_NO_THREAD_SAFETY_ANALYSIS \
  TF_INTERNAL_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis)

// TF_TS_UNCHECKED should be placed around lock expressions that are not valid
// C++ syntax, but which are present for documentation purposes.  These
// annotations will be ignored by the analysis.
#define TF_TS_UNCHECKED(x) ""

#endif  // TENSORFLOW_TSL_PLATFORM_THREAD_ANNOTATIONS_H_
