// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN2_MEMORY_H
#define EIGEN2_MEMORY_H

namespace Eigen { 

inline void* ei_aligned_malloc(size_t size) { return internal::aligned_malloc(size); }
inline void  ei_aligned_free(void *ptr) { internal::aligned_free(ptr); }
inline void* ei_aligned_realloc(void *ptr, size_t new_size, size_t old_size) { return internal::aligned_realloc(ptr, new_size, old_size); }
inline void* ei_handmade_aligned_malloc(size_t size) { return internal::handmade_aligned_malloc(size); }
inline void  ei_handmade_aligned_free(void *ptr) { internal::handmade_aligned_free(ptr); }

template<bool Align> inline void* ei_conditional_aligned_malloc(size_t size)
{
  return internal::conditional_aligned_malloc<Align>(size);
}
template<bool Align> inline void ei_conditional_aligned_free(void *ptr)
{
  internal::conditional_aligned_free<Align>(ptr);
}
template<bool Align> inline void* ei_conditional_aligned_realloc(void* ptr, size_t new_size, size_t old_size)
{
  return internal::conditional_aligned_realloc<Align>(ptr, new_size, old_size);
}

template<typename T> inline T* ei_aligned_new(size_t size)
{
  return internal::aligned_new<T>(size);
}
template<typename T> inline void ei_aligned_delete(T *ptr, size_t size)
{
  return internal::aligned_delete(ptr, size);
}

} // end namespace Eigen

#endif // EIGEN2_MACROS_H
