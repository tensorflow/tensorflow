// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Kenneth Riddile <kfriddile@yahoo.com>
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
// Copyright (C) 2010 Thomas Capricelli <orzel@freehackers.org>
// Copyright (C) 2013 Pavel Holoborodko <pavel@holoborodko.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


/*****************************************************************************
*** Platform checks for aligned malloc functions                           ***
*****************************************************************************/

#ifndef EIGEN_MEMORY_H
#define EIGEN_MEMORY_H

// See bug 554 (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=554)
// It seems to be unsafe to check _POSIX_ADVISORY_INFO without including unistd.h first.
// Currently, let's include it only on unix systems:
#if defined(__unix__) || defined(__unix)
  #include <unistd.h>
  #if ((defined __QNXNTO__) || (defined _GNU_SOURCE) || ((defined _XOPEN_SOURCE) && (_XOPEN_SOURCE >= 600))) && (defined _POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO > 0)
    #define EIGEN_HAS_POSIX_MEMALIGN 1
  #endif
#endif

#ifndef EIGEN_HAS_POSIX_MEMALIGN
  #define EIGEN_HAS_POSIX_MEMALIGN 0
#endif

#if defined EIGEN_VECTORIZE_SSE || defined EIGEN_VECTORIZE_AVX
  #define EIGEN_HAS_MM_MALLOC 1
#else
  #define EIGEN_HAS_MM_MALLOC 0
#endif

namespace Eigen {

namespace internal {

EIGEN_DEVICE_FUNC inline void throw_std_bad_alloc()
{
#ifndef __CUDA_ARCH__
  #ifdef EIGEN_EXCEPTIONS
    throw std::bad_alloc();
  #else
    std::size_t huge = static_cast<std::size_t>(-1);
    new int[huge];
  #endif
#endif
}

/*****************************************************************************
*** Implementation of handmade aligned functions                           ***
*****************************************************************************/

/* ----- Hand made implementations of aligned malloc/free and realloc ----- */

/** \internal Like malloc, but the returned pointer is guaranteed to be 16-byte aligned.
  * Fast, but wastes 16 additional bytes of memory. Does not throw any exception.
  */
inline void* handmade_aligned_malloc(std::size_t size)
{
  void *original = std::malloc(size+EIGEN_ALIGN_BYTES);
  if (original == 0) return 0;
  void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(original) & ~(std::size_t(EIGEN_ALIGN_BYTES-1))) + EIGEN_ALIGN_BYTES);
  *(reinterpret_cast<void**>(aligned) - 1) = original;
  return aligned;
}

/** \internal Frees memory allocated with handmade_aligned_malloc */
inline void handmade_aligned_free(void *ptr)
{
  if (ptr) std::free(*(reinterpret_cast<void**>(ptr) - 1));
}

/** \internal
  * \brief Reallocates aligned memory.
  * Since we know that our handmade version is based on std::realloc
  * we can use std::realloc to implement efficient reallocation.
  */
inline void* handmade_aligned_realloc(void* ptr, std::size_t size, std::size_t = 0)
{
  if (ptr == 0) return handmade_aligned_malloc(size);
  void *original = *(reinterpret_cast<void**>(ptr) - 1);
  std::ptrdiff_t previous_offset = static_cast<char *>(ptr)-static_cast<char *>(original);
  original = std::realloc(original,size+EIGEN_ALIGN_BYTES);
  if (original == 0) return 0;
  void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(original) & ~(std::size_t(EIGEN_ALIGN_BYTES-1))) + EIGEN_ALIGN_BYTES);
  void *previous_aligned = static_cast<char *>(original)+previous_offset;
  if(aligned!=previous_aligned)
    std::memmove(aligned, previous_aligned, size);

  *(reinterpret_cast<void**>(aligned) - 1) = original;
  return aligned;
}

/*****************************************************************************
*** Implementation of generic aligned realloc (when no realloc can be used)***
*****************************************************************************/

EIGEN_DEVICE_FUNC void* aligned_malloc(std::size_t size);
EIGEN_DEVICE_FUNC void  aligned_free(void *ptr);

/** \internal
  * \brief Reallocates aligned memory.
  * Allows reallocation with aligned ptr types. This implementation will
  * always create a new memory chunk and copy the old data.
  */
inline void* generic_aligned_realloc(void* ptr, size_t size, size_t old_size)
{
  if (ptr==0)
    return aligned_malloc(size);

  if (size==0)
  {
    aligned_free(ptr);
    return 0;
  }

  void* newptr = aligned_malloc(size);
  if (newptr == 0)
  {
    #ifdef EIGEN_HAS_ERRNO
    errno = ENOMEM; // according to the standard
    #endif
    return 0;
  }

  if (ptr != 0)
  {
    std::memcpy(newptr, ptr, (std::min)(size,old_size));
    aligned_free(ptr);
  }

  return newptr;
}

/*****************************************************************************
*** Implementation of portable aligned versions of malloc/free/realloc     ***
*****************************************************************************/

#ifdef EIGEN_NO_MALLOC
EIGEN_DEVICE_FUNC inline void check_that_malloc_is_allowed()
{
  eigen_assert(false && "heap allocation is forbidden (EIGEN_NO_MALLOC is defined)");
}
#elif defined EIGEN_RUNTIME_NO_MALLOC
EIGEN_DEVICE_FUNC inline bool is_malloc_allowed_impl(bool update, bool new_value = false)
{
  static bool value = true;
  if (update == 1)
    value = new_value;
  return value;
}
EIGEN_DEVICE_FUNC inline bool is_malloc_allowed() { return is_malloc_allowed_impl(false); }
EIGEN_DEVICE_FUNC inline bool set_is_malloc_allowed(bool new_value) { return is_malloc_allowed_impl(true, new_value); }
EIGEN_DEVICE_FUNC inline void check_that_malloc_is_allowed()
{
  eigen_assert(is_malloc_allowed() && "heap allocation is forbidden (EIGEN_RUNTIME_NO_MALLOC is defined and g_is_malloc_allowed is false)");
}
#else
EIGEN_DEVICE_FUNC inline void check_that_malloc_is_allowed()
{}
#endif

/** \internal Allocates \a size bytes. The returned pointer is guaranteed to have 16 or 32 bytes alignment depending on the requirements.
  * On allocation error, the returned pointer is null, and std::bad_alloc is thrown.
  */
EIGEN_DEVICE_FUNC
inline void* aligned_malloc(size_t size)
{
  check_that_malloc_is_allowed();

  void *result;
  #if !EIGEN_ALIGN
    result = std::malloc(size);
  #elif EIGEN_HAS_POSIX_MEMALIGN
    if(posix_memalign(&result, EIGEN_ALIGN_BYTES, size)) result = 0;
  #elif EIGEN_HAS_MM_MALLOC
    result = _mm_malloc(size, EIGEN_ALIGN_BYTES);
  #elif defined(_MSC_VER) && (!defined(_WIN32_WCE))
    result = _aligned_malloc(size, EIGEN_ALIGN_BYTES);
  #else
    result = handmade_aligned_malloc(size);
  #endif

  if(!result && size)
    throw_std_bad_alloc();

  return result;
}

/** \internal Frees memory allocated with aligned_malloc. */
EIGEN_DEVICE_FUNC
inline void aligned_free(void *ptr)
{
  #if !EIGEN_ALIGN
    std::free(ptr);
  #elif EIGEN_HAS_POSIX_MEMALIGN
    std::free(ptr);
  #elif EIGEN_HAS_MM_MALLOC
    _mm_free(ptr);
  #elif defined(_MSC_VER) && (!defined(_WIN32_WCE))
    _aligned_free(ptr);
  #else
    handmade_aligned_free(ptr);
  #endif
}

/**
* \internal
* \brief Reallocates an aligned block of memory.
* \throws std::bad_alloc on allocation failure
**/
inline void* aligned_realloc(void *ptr, size_t new_size, size_t old_size)
{
  EIGEN_UNUSED_VARIABLE(old_size);

  void *result;
#if !EIGEN_ALIGN
  result = std::realloc(ptr,new_size);
#elif EIGEN_HAS_POSIX_MEMALIGN
  result = generic_aligned_realloc(ptr,new_size,old_size);
#elif EIGEN_HAS_MM_MALLOC
  // The defined(_mm_free) is just here to verify that this MSVC version
  // implements _mm_malloc/_mm_free based on the corresponding _aligned_
  // functions. This may not always be the case and we just try to be safe.
  #if EIGEN_OS_WIN_STRICT && defined(_mm_free)
    result = _aligned_realloc(ptr,new_size,EIGEN_ALIGN_BYTES);
  #else
    result = generic_aligned_realloc(ptr,new_size,old_size);
  #endif
#elif EIGEN_OS_WIN_STRICT
  result = _aligned_realloc(ptr,new_size,EIGEN_ALIGN_BYTES);
#else
  result = handmade_aligned_realloc(ptr,new_size,old_size);
#endif

  if (!result && new_size)
    throw_std_bad_alloc();

  return result;
}

/*****************************************************************************
*** Implementation of conditionally aligned functions                      ***
*****************************************************************************/

/** \internal Allocates \a size bytes. If Align is true, then the returned ptr is 16-byte-aligned.
  * On allocation error, the returned pointer is null, and a std::bad_alloc is thrown.
  */
template<bool Align> EIGEN_DEVICE_FUNC inline void* conditional_aligned_malloc(size_t size)
{
  return aligned_malloc(size);
}

template<> EIGEN_DEVICE_FUNC inline void* conditional_aligned_malloc<false>(size_t size)
{
  check_that_malloc_is_allowed();

  void *result = std::malloc(size);
  if(!result && size)
    throw_std_bad_alloc();
  return result;
}

/** \internal Frees memory allocated with conditional_aligned_malloc */
template<bool Align> EIGEN_DEVICE_FUNC inline void conditional_aligned_free(void *ptr)
{
  aligned_free(ptr);
}

template<> EIGEN_DEVICE_FUNC inline void conditional_aligned_free<false>(void *ptr)
{
  std::free(ptr);
}

template<bool Align> inline void* conditional_aligned_realloc(void* ptr, size_t new_size, size_t old_size)
{
  return aligned_realloc(ptr, new_size, old_size);
}

template<> inline void* conditional_aligned_realloc<false>(void* ptr, size_t new_size, size_t)
{
  return std::realloc(ptr, new_size);
}

/*****************************************************************************
*** Construction/destruction of array elements                             ***
*****************************************************************************/

/** \internal Constructs the elements of an array.
  * The \a size parameter tells on how many objects to call the constructor of T.
  */
template<typename T> EIGEN_DEVICE_FUNC inline T* construct_elements_of_array(T *ptr, size_t size)
{
  for (size_t i=0; i < size; ++i) ::new (ptr + i) T;
  return ptr;
}

/** \internal Destructs the elements of an array.
  * The \a size parameters tells on how many objects to call the destructor of T.
  */
template<typename T> EIGEN_DEVICE_FUNC inline void destruct_elements_of_array(T *ptr, size_t size)
{
  // always destruct an array starting from the end.
  if(ptr)
    while(size) ptr[--size].~T();
}

/*****************************************************************************
*** Implementation of aligned new/delete-like functions                    ***
*****************************************************************************/

template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void check_size_for_overflow(size_t size)
{
  if(size > size_t(-1) / sizeof(T))
    throw_std_bad_alloc();
}

/** \internal Allocates \a size objects of type T. The returned pointer is guaranteed to have 16 bytes alignment.
  * On allocation error, the returned pointer is undefined, but a std::bad_alloc is thrown.
  * The default constructor of T is called.
  */
template<typename T> EIGEN_DEVICE_FUNC inline T* aligned_new(size_t size)
{
  check_size_for_overflow<T>(size);
  T *result = reinterpret_cast<T*>(aligned_malloc(sizeof(T)*size));
  return construct_elements_of_array(result, size);
}

template<typename T, bool Align> EIGEN_DEVICE_FUNC inline T* conditional_aligned_new(size_t size)
{
  check_size_for_overflow<T>(size);
  T *result = reinterpret_cast<T*>(conditional_aligned_malloc<Align>(sizeof(T)*size));
  return construct_elements_of_array(result, size);
}

template<typename T> EIGEN_DEVICE_FUNC inline T* allocate_uvm(size_t size)
{
#if defined(EIGEN_USE_GPU) && defined(__CUDA_ARCH__)
  return (T*)malloc(size);
#elif defined(EIGEN_USE_GPU) && defined(__NVCC__)
  T* result = NULL;
  if (cudaMallocManaged(&result, size) != cudaSuccess) {
    throw_std_bad_alloc();
  }
  return result;
#else
  return reinterpret_cast<T*>(conditional_aligned_malloc<true>(sizeof(T)*size));
#endif
}

template<typename T> EIGEN_DEVICE_FUNC void deallocate_uvm(T* ptr)
{
#if defined(EIGEN_USE_GPU) && defined(__CUDA_ARCH__)
  free(ptr);
#elif defined(EIGEN_USE_GPU) && defined(__NVCC__)
  if (cudaFree(ptr) != cudaSuccess) {
    throw_std_bad_alloc();
  }
#else
  return conditional_aligned_free<true>(ptr);
#endif
}

/** \internal Deletes objects constructed with aligned_new
  * The \a size parameters tells on how many objects to call the destructor of T.
  */
template<typename T> EIGEN_DEVICE_FUNC  inline void aligned_delete(T *ptr, size_t size)
{
  destruct_elements_of_array<T>(ptr, size);
  aligned_free(ptr);
}

/** \internal Deletes objects constructed with conditional_aligned_new
  * The \a size parameters tells on how many objects to call the destructor of T.
  */
template<typename T, bool Align> EIGEN_DEVICE_FUNC inline void conditional_aligned_delete(T *ptr, size_t size)
{
  destruct_elements_of_array<T>(ptr, size);
  conditional_aligned_free<Align>(ptr);
}

template<typename T, bool Align> EIGEN_DEVICE_FUNC inline T* conditional_aligned_realloc_new(T* pts, size_t new_size, size_t old_size)
{
  check_size_for_overflow<T>(new_size);
  check_size_for_overflow<T>(old_size);
  if(new_size < old_size)
    destruct_elements_of_array(pts+new_size, old_size-new_size);
  T *result = reinterpret_cast<T*>(conditional_aligned_realloc<Align>(reinterpret_cast<void*>(pts), sizeof(T)*new_size, sizeof(T)*old_size));
  if(new_size > old_size)
    construct_elements_of_array(result+old_size, new_size-old_size);
  return result;
}


template<typename T, bool Align> EIGEN_DEVICE_FUNC inline T* conditional_aligned_new_auto(size_t size)
{
  check_size_for_overflow<T>(size);
  T *result = reinterpret_cast<T*>(conditional_aligned_malloc<Align>(sizeof(T)*size));
  if(NumTraits<T>::RequireInitialization)
    construct_elements_of_array(result, size);
  return result;
}

template<typename T, bool Align, bool UseUVM> EIGEN_DEVICE_FUNC inline T* conditional_managed_new_auto(size_t size)
{
  check_size_for_overflow<T>(size);
  T *result;
  if (UseUVM) {
    result = allocate_uvm<T>(size*sizeof(T));
  }
  else {
    result = reinterpret_cast<T*>(conditional_aligned_malloc<Align>(sizeof(T)*size));
  }
  if(NumTraits<T>::RequireInitialization)
    construct_elements_of_array(result, size);
  return result;
}

template<typename T, bool Align, bool UseUVM> EIGEN_DEVICE_FUNC inline void conditional_managed_delete_auto(T* ptr, size_t size)
{
  if(NumTraits<T>::RequireInitialization)
    destruct_elements_of_array<T>(ptr, size);
  if (UseUVM) {
    deallocate_uvm(ptr);
  }
  else {
    conditional_aligned_free<Align>(ptr);
  }
}

template<typename T, bool Align> inline T* conditional_aligned_realloc_new_auto(T* pts, size_t new_size, size_t old_size)
{
  check_size_for_overflow<T>(new_size);
  check_size_for_overflow<T>(old_size);
  if(NumTraits<T>::RequireInitialization && (new_size < old_size))
    destruct_elements_of_array(pts+new_size, old_size-new_size);
  T *result = reinterpret_cast<T*>(conditional_aligned_realloc<Align>(reinterpret_cast<void*>(pts), sizeof(T)*new_size, sizeof(T)*old_size));
  if(NumTraits<T>::RequireInitialization && (new_size > old_size))
    construct_elements_of_array(result+old_size, new_size-old_size);
  return result;
}

template<typename T, bool Align> EIGEN_DEVICE_FUNC inline void conditional_aligned_delete_auto(T *ptr, size_t size)
{
  if(NumTraits<T>::RequireInitialization)
    destruct_elements_of_array<T>(ptr, size);
  conditional_aligned_free<Align>(ptr);
}

/****************************************************************************/

/** \internal Returns the index of the first element of the array that is well aligned for vectorization.
  *
  * \param array the address of the start of the array
  * \param size the size of the array
  *
  * \note If no element of the array is well aligned, the size of the array is returned. Typically,
  * for example with SSE, "well aligned" means 16-byte-aligned. If vectorization is disabled or if the
  * packet size for the given scalar type is 1, then everything is considered well-aligned.
  *
  * \note If the scalar type is vectorizable, we rely on the following assumptions: sizeof(Scalar) is a
  * power of 2, the packet size in bytes is also a power of 2, and is a multiple of sizeof(Scalar). On the
  * other hand, we do not assume that the array address is a multiple of sizeof(Scalar), as that fails for
  * example with Scalar=double on certain 32-bit platforms, see bug #79.
  *
  * There is also the variant first_aligned(const MatrixBase&) defined in DenseCoeffsBase.h.
  */
template<typename Scalar, typename Index>
inline Index first_aligned(const Scalar* array, Index size)
{
  enum { PacketSize = packet_traits<Scalar>::size,
         PacketAlignedMask = PacketSize-1
  };

  if(PacketSize==1)
  {
    // Either there is no vectorization, or a packet consists of exactly 1 scalar so that all elements
    // of the array have the same alignment.
    return 0;
  }
  else if(size_t(array) & (sizeof(Scalar)-1))
  {
    // There is vectorization for this scalar type, but the array is not aligned to the size of a single scalar.
    // Consequently, no element of the array is well aligned.
    return size;
  }
  else
  {
    return std::min<Index>( (PacketSize - (Index((size_t(array)/sizeof(Scalar))) & PacketAlignedMask))
                           & PacketAlignedMask, size);
  }
}

/** \internal Returns the smallest integer multiple of \a base and greater or equal to \a size
  */
template<typename Index>
inline Index first_multiple(Index size, Index base)
{
  return ((size+base-1)/base)*base;
}

// std::copy is much slower than memcpy, so let's introduce a smart_copy which
// use memcpy on trivial types, i.e., on types that does not require an initialization ctor.
template<typename T, bool UseMemcpy> struct smart_copy_helper;

template<typename T> EIGEN_DEVICE_FUNC void smart_copy(const T* start, const T* end, T* target)
{
  smart_copy_helper<T,!NumTraits<T>::RequireInitialization>::run(start, end, target);
}

template<typename T> struct smart_copy_helper<T,true> {
  static inline EIGEN_DEVICE_FUNC void run(const T* start, const T* end, T* target)
  { memcpy(target, start, std::ptrdiff_t(end)-std::ptrdiff_t(start)); }
};

template<typename T> struct smart_copy_helper<T,false> {
  static inline EIGEN_DEVICE_FUNC void run(const T* start, const T* end, T* target)
  { std::copy(start, end, target); }
};

// intelligent memmove. falls back to std::memmove for POD types, uses std::copy otherwise.
template<typename T, bool UseMemmove> struct smart_memmove_helper;

template<typename T> void smart_memmove(const T* start, const T* end, T* target)
{
    smart_memmove_helper<T,!NumTraits<T>::RequireInitialization>::run(start, end, target);
}

template<typename T> struct smart_memmove_helper<T,true> {
    static inline void run(const T* start, const T* end, T* target)
    { std::memmove(target, start, std::ptrdiff_t(end)-std::ptrdiff_t(start)); }
};

template<typename T> struct smart_memmove_helper<T,false> {
    static inline void run(const T* start, const T* end, T* target)
    {
        if (uintptr_t(target) < uintptr_t(start))
        {
            std::copy(start, end, target);
        }
        else
        {
            std::ptrdiff_t count = (std::ptrdiff_t(end)-std::ptrdiff_t(start)) / sizeof(T);
            std::copy_backward(start, end, target + count);
        }
    }
};


/*****************************************************************************
*** Implementation of runtime stack allocation (falling back to malloc)    ***
*****************************************************************************/

// you can overwrite Eigen's default behavior regarding alloca by defining EIGEN_ALLOCA
// to the appropriate stack allocation function
#ifndef EIGEN_ALLOCA
  #if (defined __linux__) || (defined __APPLE__)
    #define EIGEN_ALLOCA alloca
  #elif defined(_MSC_VER)
    #define EIGEN_ALLOCA _alloca
  #endif
#endif

// This helper class construct the allocated memory, and takes care of destructing and freeing the handled data
// at destruction time. In practice this helper class is mainly useful to avoid memory leak in case of exceptions.
template<typename T> class aligned_stack_memory_handler
{
  public:
    /* Creates a stack_memory_handler responsible for the buffer \a ptr of size \a size.
     * Note that \a ptr can be 0 regardless of the other parameters.
     * This constructor takes care of constructing/initializing the elements of the buffer if required by the scalar type T (see NumTraits<T>::RequireInitialization).
     * In this case, the buffer elements will also be destructed when this handler will be destructed.
     * Finally, if \a dealloc is true, then the pointer \a ptr is freed.
     **/
    aligned_stack_memory_handler(T* ptr, size_t size, bool dealloc)
      : m_ptr(ptr), m_size(size), m_deallocate(dealloc)
    {
      if(NumTraits<T>::RequireInitialization && m_ptr)
        Eigen::internal::construct_elements_of_array(m_ptr, size);
    }
    ~aligned_stack_memory_handler()
    {
      if(NumTraits<T>::RequireInitialization && m_ptr)
        Eigen::internal::destruct_elements_of_array<T>(m_ptr, m_size);
      if(m_deallocate)
        Eigen::internal::aligned_free(m_ptr);
    }
  protected:
    T* m_ptr;
    size_t m_size;
    bool m_deallocate;
};

} // end namespace internal

/** \internal
  * Declares, allocates and construct an aligned buffer named NAME of SIZE elements of type TYPE on the stack
  * if SIZE is smaller than EIGEN_STACK_ALLOCATION_LIMIT, and if stack allocation is supported by the platform
  * (currently, this is Linux and Visual Studio only). Otherwise the memory is allocated on the heap.
  * The allocated buffer is automatically deleted when exiting the scope of this declaration.
  * If BUFFER is non null, then the declared variable is simply an alias for BUFFER, and no allocation/deletion occurs.
  * Here is an example:
  * \code
  * {
  *   ei_declare_aligned_stack_constructed_variable(float,data,size,0);
  *   // use data[0] to data[size-1]
  * }
  * \endcode
  * The underlying stack allocation function can controlled with the EIGEN_ALLOCA preprocessor token.
  */
#ifdef EIGEN_ALLOCA
  // The native alloca() that comes with llvm aligns buffer on 16 bytes even when AVX is enabled.
#if defined(__arm__) || defined(_WIN32) || EIGEN_ALIGN_BYTES > 16
    #define EIGEN_ALIGNED_ALLOCA(SIZE) reinterpret_cast<void*>((reinterpret_cast<size_t>(EIGEN_ALLOCA(SIZE+EIGEN_ALIGN_BYTES)) & ~(size_t(EIGEN_ALIGN_BYTES-1))) + EIGEN_ALIGN_BYTES)
  #else
    #define EIGEN_ALIGNED_ALLOCA EIGEN_ALLOCA
  #endif

  #define ei_declare_aligned_stack_constructed_variable(TYPE,NAME,SIZE,BUFFER) \
    Eigen::internal::check_size_for_overflow<TYPE>(SIZE); \
    TYPE* NAME = (BUFFER)!=0 ? (BUFFER) \
               : reinterpret_cast<TYPE*>( \
                      (sizeof(TYPE)*SIZE<=EIGEN_STACK_ALLOCATION_LIMIT) ? EIGEN_ALIGNED_ALLOCA(sizeof(TYPE)*SIZE) \
                    : Eigen::internal::aligned_malloc(sizeof(TYPE)*SIZE) );  \
    Eigen::internal::aligned_stack_memory_handler<TYPE> EIGEN_CAT(NAME,_stack_memory_destructor)((BUFFER)==0 ? NAME : 0,SIZE,sizeof(TYPE)*SIZE>EIGEN_STACK_ALLOCATION_LIMIT)

#else

  #define ei_declare_aligned_stack_constructed_variable(TYPE,NAME,SIZE,BUFFER) \
    Eigen::internal::check_size_for_overflow<TYPE>(SIZE); \
    TYPE* NAME = (BUFFER)!=0 ? BUFFER : reinterpret_cast<TYPE*>(Eigen::internal::aligned_malloc(sizeof(TYPE)*SIZE));    \
    Eigen::internal::aligned_stack_memory_handler<TYPE> EIGEN_CAT(NAME,_stack_memory_destructor)((BUFFER)==0 ? NAME : 0,SIZE,true)

#endif


/*****************************************************************************
*** Implementation of EIGEN_MAKE_ALIGNED_OPERATOR_NEW [_IF]                ***
*****************************************************************************/

#if EIGEN_ALIGN
  #ifdef EIGEN_EXCEPTIONS
    #define EIGEN_MAKE_ALIGNED_OPERATOR_NEW_NOTHROW(NeedsToAlign) \
      void* operator new(size_t size, const std::nothrow_t&) throw() { \
        try { return Eigen::internal::conditional_aligned_malloc<NeedsToAlign>(size); } \
        catch (...) { return 0; } \
        return 0; \
      }
  #else
    #define EIGEN_MAKE_ALIGNED_OPERATOR_NEW_NOTHROW(NeedsToAlign) \
      void* operator new(size_t size, const std::nothrow_t&) throw() { \
        return Eigen::internal::conditional_aligned_malloc<NeedsToAlign>(size); \
      }
  #endif

  #define EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign) \
      void *operator new(size_t size) { \
        return Eigen::internal::conditional_aligned_malloc<NeedsToAlign>(size); \
      } \
      void *operator new[](size_t size) { \
        return Eigen::internal::conditional_aligned_malloc<NeedsToAlign>(size); \
      } \
      void operator delete(void * ptr) throw() { Eigen::internal::conditional_aligned_free<NeedsToAlign>(ptr); } \
      void operator delete[](void * ptr) throw() { Eigen::internal::conditional_aligned_free<NeedsToAlign>(ptr); } \
      /* in-place new and delete. since (at least afaik) there is no actual   */ \
      /* memory allocated we can safely let the default implementation handle */ \
      /* this particular case. */ \
      static void *operator new(size_t size, void *ptr) { return ::operator new(size,ptr); } \
      static void *operator new[](size_t size, void* ptr) { return ::operator new[](size,ptr); } \
      void operator delete(void * memory, void *ptr) throw() { return ::operator delete(memory,ptr); } \
      void operator delete[](void * memory, void *ptr) throw() { return ::operator delete[](memory,ptr); } \
      /* nothrow-new (returns zero instead of std::bad_alloc) */ \
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW_NOTHROW(NeedsToAlign) \
      void operator delete(void *ptr, const std::nothrow_t&) throw() { \
        Eigen::internal::conditional_aligned_free<NeedsToAlign>(ptr); \
      } \
      typedef void eigen_aligned_operator_new_marker_type;
#else
  #define EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
#endif

#define EIGEN_MAKE_ALIGNED_OPERATOR_NEW EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(true)
#define EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(Scalar,Size) \
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(bool(((Size)!=Eigen::Dynamic) && ((sizeof(Scalar)*(Size))%EIGEN_ALIGN_BYTES==0)))

/****************************************************************************/

/** \class aligned_allocator
* \ingroup Core_Module
*
* \brief STL compatible allocator to use with with 16 byte aligned types
*
* Example:
* \code
* // Matrix4f requires 16 bytes alignment:
* std::map< int, Matrix4f, std::less<int>,
*           aligned_allocator<std::pair<const int, Matrix4f> > > my_map_mat4;
* // Vector3f does not require 16 bytes alignment, no need to use Eigen's allocator:
* std::map< int, Vector3f > my_map_vec3;
* \endcode
*
* \sa \ref TopicStlContainers.
*/
template<class T>
class aligned_allocator : public std::allocator<T>
{
public:
  typedef size_t          size_type;
  typedef std::ptrdiff_t  difference_type;
  typedef T*              pointer;
  typedef const T*        const_pointer;
  typedef T&              reference;
  typedef const T&        const_reference;
  typedef T               value_type;

  template<class U>
  struct rebind
  {
    typedef aligned_allocator<U> other;
  };

  aligned_allocator() : std::allocator<T>() {}

  aligned_allocator(const aligned_allocator& other) : std::allocator<T>(other) {}

  template<class U>
  aligned_allocator(const aligned_allocator<U>& other) : std::allocator<T>(other) {}

  ~aligned_allocator() {}

  pointer allocate(size_type num, const void* /*hint*/ = 0)
  {
    internal::check_size_for_overflow<T>(num);
    return static_cast<pointer>( internal::aligned_malloc(num * sizeof(T)) );
  }

  void deallocate(pointer p, size_type /*num*/)
  {
    internal::aligned_free(p);
  }
};

//---------- Cache sizes ----------

#if !defined(EIGEN_NO_CPUID)
#  if EIGEN_COMP_GNUC && EIGEN_ARCH_i386_OR_x86_64
#    if defined(__PIC__) && EIGEN_ARCH_i386
       // Case for x86 with PIC
#      define EIGEN_CPUID(abcd,func,id) \
         __asm__ __volatile__ ("xchgl %%ebx, %k1;cpuid; xchgl %%ebx,%k1": "=a" (abcd[0]), "=&r" (abcd[1]), "=c" (abcd[2]), "=d" (abcd[3]) : "a" (func), "c" (id));
#    elif defined(__PIC__) && EIGEN_ARCH_x86_64
       // Case for x64 with PIC. In theory this is only a problem with recent gcc and with medium or large code model, not with the default small code model.
       // However, we cannot detect which code model is used, and the xchg overhead is negligible anyway.
#      define EIGEN_CPUID(abcd,func,id) \
        __asm__ __volatile__ ("xchg{q}\t{%%}rbx, %q1; cpuid; xchg{q}\t{%%}rbx, %q1": "=a" (abcd[0]), "=&r" (abcd[1]), "=c" (abcd[2]), "=d" (abcd[3]) : "0" (func), "2" (id));
#    else
       // Case for x86_64 or x86 w/o PIC
#      define EIGEN_CPUID(abcd,func,id) \
         __asm__ __volatile__ ("cpuid": "=a" (abcd[0]), "=b" (abcd[1]), "=c" (abcd[2]), "=d" (abcd[3]) : "0" (func), "2" (id) );
#    endif
#  elif EIGEN_COMP_MSVC
#    if (EIGEN_COMP_MSVC > 1500) && EIGEN_ARCH_i386_OR_x86_64
#      define EIGEN_CPUID(abcd,func,id) __cpuidex((int*)abcd,func,id)
#    endif
#  endif
#endif

namespace internal {

#ifdef EIGEN_CPUID

inline bool cpuid_is_vendor(int abcd[4], const char* vendor)
{
  return abcd[1]==(reinterpret_cast<const int*>(vendor))[0] && abcd[3]==(reinterpret_cast<const int*>(vendor))[1] && abcd[2]==(reinterpret_cast<const int*>(vendor))[2];
}

inline void queryCacheSizes_intel_direct(int& l1, int& l2, int& l3)
{
  int abcd[4];
  l1 = l2 = l3 = 0;
  int cache_id = 0;
  int cache_type = 0;
  do {
    abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
    EIGEN_CPUID(abcd,0x4,cache_id);
    cache_type  = (abcd[0] & 0x0F) >> 0;
    if(cache_type==1||cache_type==3) // data or unified cache
    {
      int cache_level = (abcd[0] & 0xE0) >> 5;  // A[7:5]
      int ways        = (abcd[1] & 0xFFC00000) >> 22; // B[31:22]
      int partitions  = (abcd[1] & 0x003FF000) >> 12; // B[21:12]
      int line_size   = (abcd[1] & 0x00000FFF) >>  0; // B[11:0]
      int sets        = (abcd[2]);                    // C[31:0]

      int cache_size = (ways+1) * (partitions+1) * (line_size+1) * (sets+1);

      switch(cache_level)
      {
        case 1: l1 = cache_size; break;
        case 2: l2 = cache_size; break;
        case 3: l3 = cache_size; break;
        default: break;
      }
    }
    cache_id++;
  } while(cache_type>0 && cache_id<16);
}

inline void queryCacheSizes_intel_codes(int& l1, int& l2, int& l3)
{
  int abcd[4];
  abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
  l1 = l2 = l3 = 0;
  EIGEN_CPUID(abcd,0x00000002,0);
  unsigned char * bytes = reinterpret_cast<unsigned char *>(abcd)+2;
  bool check_for_p2_core2 = false;
  for(int i=0; i<14; ++i)
  {
    switch(bytes[i])
    {
      case 0x0A: l1 = 8; break;   // 0Ah   data L1 cache, 8 KB, 2 ways, 32 byte lines
      case 0x0C: l1 = 16; break;  // 0Ch   data L1 cache, 16 KB, 4 ways, 32 byte lines
      case 0x0E: l1 = 24; break;  // 0Eh   data L1 cache, 24 KB, 6 ways, 64 byte lines
      case 0x10: l1 = 16; break;  // 10h   data L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
      case 0x15: l1 = 16; break;  // 15h   code L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
      case 0x2C: l1 = 32; break;  // 2Ch   data L1 cache, 32 KB, 8 ways, 64 byte lines
      case 0x30: l1 = 32; break;  // 30h   code L1 cache, 32 KB, 8 ways, 64 byte lines
      case 0x60: l1 = 16; break;  // 60h   data L1 cache, 16 KB, 8 ways, 64 byte lines, sectored
      case 0x66: l1 = 8; break;   // 66h   data L1 cache, 8 KB, 4 ways, 64 byte lines, sectored
      case 0x67: l1 = 16; break;  // 67h   data L1 cache, 16 KB, 4 ways, 64 byte lines, sectored
      case 0x68: l1 = 32; break;  // 68h   data L1 cache, 32 KB, 4 ways, 64 byte lines, sectored
      case 0x1A: l2 = 96; break;   // code and data L2 cache, 96 KB, 6 ways, 64 byte lines (IA-64)
      case 0x22: l3 = 512; break;   // code and data L3 cache, 512 KB, 4 ways (!), 64 byte lines, dual-sectored
      case 0x23: l3 = 1024; break;   // code and data L3 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
      case 0x25: l3 = 2048; break;   // code and data L3 cache, 2048 KB, 8 ways, 64 byte lines, dual-sectored
      case 0x29: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 8 ways, 64 byte lines, dual-sectored
      case 0x39: l2 = 128; break;   // code and data L2 cache, 128 KB, 4 ways, 64 byte lines, sectored
      case 0x3A: l2 = 192; break;   // code and data L2 cache, 192 KB, 6 ways, 64 byte lines, sectored
      case 0x3B: l2 = 128; break;   // code and data L2 cache, 128 KB, 2 ways, 64 byte lines, sectored
      case 0x3C: l2 = 256; break;   // code and data L2 cache, 256 KB, 4 ways, 64 byte lines, sectored
      case 0x3D: l2 = 384; break;   // code and data L2 cache, 384 KB, 6 ways, 64 byte lines, sectored
      case 0x3E: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 64 byte lines, sectored
      case 0x40: l2 = 0; break;   // no integrated L2 cache (P6 core) or L3 cache (P4 core)
      case 0x41: l2 = 128; break;   // code and data L2 cache, 128 KB, 4 ways, 32 byte lines
      case 0x42: l2 = 256; break;   // code and data L2 cache, 256 KB, 4 ways, 32 byte lines
      case 0x43: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 32 byte lines
      case 0x44: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 4 ways, 32 byte lines
      case 0x45: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 4 ways, 32 byte lines
      case 0x46: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines
      case 0x47: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 8 ways, 64 byte lines
      case 0x48: l2 = 3072; break;   // code and data L2 cache, 3072 KB, 12 ways, 64 byte lines
      case 0x49: if(l2!=0) l3 = 4096; else {check_for_p2_core2=true; l3 = l2 = 4096;} break;// code and data L3 cache, 4096 KB, 16 ways, 64 byte lines (P4) or L2 for core2
      case 0x4A: l3 = 6144; break;   // code and data L3 cache, 6144 KB, 12 ways, 64 byte lines
      case 0x4B: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 16 ways, 64 byte lines
      case 0x4C: l3 = 12288; break;   // code and data L3 cache, 12288 KB, 12 ways, 64 byte lines
      case 0x4D: l3 = 16384; break;   // code and data L3 cache, 16384 KB, 16 ways, 64 byte lines
      case 0x4E: l2 = 6144; break;   // code and data L2 cache, 6144 KB, 24 ways, 64 byte lines
      case 0x78: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 4 ways, 64 byte lines
      case 0x79: l2 = 128; break;   // code and data L2 cache, 128 KB, 8 ways, 64 byte lines, dual-sectored
      case 0x7A: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 64 byte lines, dual-sectored
      case 0x7B: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 64 byte lines, dual-sectored
      case 0x7C: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
      case 0x7D: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 8 ways, 64 byte lines
      case 0x7E: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 128 byte lines, sect. (IA-64)
      case 0x7F: l2 = 512; break;   // code and data L2 cache, 512 KB, 2 ways, 64 byte lines
      case 0x80: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 64 byte lines
      case 0x81: l2 = 128; break;   // code and data L2 cache, 128 KB, 8 ways, 32 byte lines
      case 0x82: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 32 byte lines
      case 0x83: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 32 byte lines
      case 0x84: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 32 byte lines
      case 0x85: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 8 ways, 32 byte lines
      case 0x86: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 64 byte lines
      case 0x87: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines
      case 0x88: l3 = 2048; break;   // code and data L3 cache, 2048 KB, 4 ways, 64 byte lines (IA-64)
      case 0x89: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines (IA-64)
      case 0x8A: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 4 ways, 64 byte lines (IA-64)
      case 0x8D: l3 = 3072; break;   // code and data L3 cache, 3072 KB, 12 ways, 128 byte lines (IA-64)

      default: break;
    }
  }
  if(check_for_p2_core2 && l2 == l3)
    l3 = 0;
  l1 *= 1024;
  l2 *= 1024;
  l3 *= 1024;
}

inline void queryCacheSizes_intel(int& l1, int& l2, int& l3, int max_std_funcs)
{
  if(max_std_funcs>=4)
    queryCacheSizes_intel_direct(l1,l2,l3);
  else
    queryCacheSizes_intel_codes(l1,l2,l3);
}

inline void queryCacheSizes_amd(int& l1, int& l2, int& l3)
{
  int abcd[4];
  abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
  EIGEN_CPUID(abcd,0x80000005,0);
  l1 = (abcd[2] >> 24) * 1024; // C[31:24] = L1 size in KB
  abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
  EIGEN_CPUID(abcd,0x80000006,0);
  l2 = (abcd[2] >> 16) * 1024; // C[31;16] = l2 cache size in KB
  l3 = ((abcd[3] & 0xFFFC000) >> 18) * 512 * 1024; // D[31;18] = l3 cache size in 512KB
}
#endif

/** \internal
 * Queries and returns the cache sizes in Bytes of the L1, L2, and L3 data caches respectively */
inline void queryCacheSizes(int& l1, int& l2, int& l3)
{
  #ifdef EIGEN_CPUID
  int abcd[4];

  // identify the CPU vendor
  EIGEN_CPUID(abcd,0x0,0);
  int max_std_funcs = abcd[1];
  if(cpuid_is_vendor(abcd,"GenuineIntel"))
    queryCacheSizes_intel(l1,l2,l3,max_std_funcs);
  else if(cpuid_is_vendor(abcd,"AuthenticAMD") || cpuid_is_vendor(abcd,"AMDisbetter!"))
    queryCacheSizes_amd(l1,l2,l3);
  else
    // by default let's use Intel's API
    queryCacheSizes_intel(l1,l2,l3,max_std_funcs);

  // here is the list of other vendors:
//   ||cpuid_is_vendor(abcd,"VIA VIA VIA ")
//   ||cpuid_is_vendor(abcd,"CyrixInstead")
//   ||cpuid_is_vendor(abcd,"CentaurHauls")
//   ||cpuid_is_vendor(abcd,"GenuineTMx86")
//   ||cpuid_is_vendor(abcd,"TransmetaCPU")
//   ||cpuid_is_vendor(abcd,"RiseRiseRise")
//   ||cpuid_is_vendor(abcd,"Geode by NSC")
//   ||cpuid_is_vendor(abcd,"SiS SiS SiS ")
//   ||cpuid_is_vendor(abcd,"UMC UMC UMC ")
//   ||cpuid_is_vendor(abcd,"NexGenDriven")
  #else
  l1 = l2 = l3 = -1;
  #endif
}

/** \internal
 * \returns the size in Bytes of the L1 data cache */
inline int queryL1CacheSize()
{
  int l1(-1), l2, l3;
  queryCacheSizes(l1,l2,l3);
  return l1;
}

inline int queryL2CacheSize()
{
  int l1, l2(-1), l3;
  queryCacheSizes(l1,l2,l3);
  return l2;
}

/** \internal
 * \returns the size in Bytes of the L2 or L3 cache if this later is present */
inline int queryTopLevelCacheSize()
{
  int l1, l2(-1), l3(-1);
  queryCacheSizes(l1,l2,l3);
  return (std::max)(l2,l3);
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_MEMORY_H
