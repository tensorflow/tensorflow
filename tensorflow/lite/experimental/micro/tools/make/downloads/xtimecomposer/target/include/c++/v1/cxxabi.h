//===--------------------------- cxxabi.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __CXXABI_H
#define __CXXABI_H 

/*
 * This header provides the interface to the C++ ABI as defined at:
 *       http://www.codesourcery.com/cxx-abi/
 */

#include <stddef.h>
#include <stdint.h>

#define _LIBCPPABI_VERSION 1001
#define LIBCXXABI_NORETURN  __attribute__((noreturn))

// FIXME: This is also in unwind.h and libunwind.h, can we consolidate?
#if !defined(__USING_SJLJ_EXCEPTIONS__) && defined(__arm__) && \
    !defined(__ARM_DWARF_EH__) && !defined(__APPLE__)
#define LIBCXXABI_ARM_EHABI 1
#else
#define LIBCXXABI_ARM_EHABI 0
#endif

#ifdef __cplusplus

namespace std {
    class type_info; // forward declaration
}


// runtime routines use C calling conventions, but are in __cxxabiv1 namespace
namespace __cxxabiv1 {  
  extern "C"  {

// 2.4.2 Allocating the Exception Object
extern void * __cxa_allocate_exception(size_t thrown_size) throw();
extern void __cxa_free_exception(void * thrown_exception) throw();

// 2.4.3 Throwing the Exception Object
// xcore requires arg 'dest' to have its fptrgroup attribute set viz:
//  __attribute__((fptrgroup("__cxa_throw_dest"))) void MyDest(void*){...}
extern LIBCXXABI_NORETURN void __cxa_throw(void * thrown_exception, 
        std::type_info * tinfo, void (*dest)(void *));

// 2.5.3 Exception Handlers
extern void * __cxa_get_exception_ptr(void * exceptionObject) throw();
extern void * __cxa_begin_catch(void * exceptionObject) throw();
extern void __cxa_end_catch();
#if LIBCXXABI_ARM_EHABI
extern bool __cxa_begin_cleanup(void * exceptionObject) throw();
extern void __cxa_end_cleanup();
#endif
extern std::type_info * __cxa_current_exception_type();

// 2.5.4 Rethrowing Exceptions
extern LIBCXXABI_NORETURN void __cxa_rethrow();



// 2.6 Auxiliary Runtime APIs
extern LIBCXXABI_NORETURN void __cxa_bad_cast(void);
extern LIBCXXABI_NORETURN void __cxa_bad_typeid(void);
extern LIBCXXABI_NORETURN void __cxa_throw_bad_array_new_length(void);



// 3.2.6 Pure Virtual Function API
extern LIBCXXABI_NORETURN void __cxa_pure_virtual(void);

// 3.2.7 Deleted Virtual Function API
extern LIBCXXABI_NORETURN void __cxa_deleted_virtual(void);

// 3.3.2 One-time Construction API
#if __arm__
extern int  __cxa_guard_acquire(uint32_t*);
extern void __cxa_guard_release(uint32_t*);
extern void __cxa_guard_abort(uint32_t*);
#else
extern int  __cxa_guard_acquire(uint64_t*);
extern void __cxa_guard_release(uint64_t*);
extern void __cxa_guard_abort(uint64_t*);
#endif

// 3.3.3 Array Construction and Destruction API
extern void* __cxa_vec_new(size_t element_count, 
                           size_t element_size, 
                           size_t padding_size, 
                           void (*constructor)(void*),
                           void (*destructor)(void*) );

extern void* __cxa_vec_new2(size_t element_count,
                            size_t element_size, 
                            size_t padding_size,
                            void  (*constructor)(void*),
                            void  (*destructor)(void*),
                            void* (*alloc)(size_t), 
                            void  (*dealloc)(void*) );

extern void* __cxa_vec_new3(size_t element_count,
                            size_t element_size, 
                            size_t padding_size,
                            void  (*constructor)(void*),
                            void  (*destructor)(void*),
                            void* (*alloc)(size_t), 
                            void  (*dealloc)(void*, size_t) );
  
extern void __cxa_vec_ctor(void*  array_address, 
                           size_t element_count,
                           size_t element_size, 
                           void (*constructor)(void*),
                           void (*destructor)(void*) );


extern void __cxa_vec_dtor(void*  array_address, 
                           size_t element_count,
                           size_t element_size, 
                           void (*destructor)(void*) );


extern void __cxa_vec_cleanup(void* array_address, 
                             size_t element_count,
                             size_t element_size, 
                             void  (*destructor)(void*) );


extern void __cxa_vec_delete(void*  array_address, 
                             size_t element_size, 
                             size_t padding_size, 
                             void  (*destructor)(void*) );


extern void __cxa_vec_delete2(void* array_address, 
                             size_t element_size, 
                             size_t padding_size, 
                             void  (*destructor)(void*),
                             void  (*dealloc)(void*) );
  

extern void __cxa_vec_delete3(void* __array_address, 
                             size_t element_size, 
                             size_t padding_size, 
                             void  (*destructor)(void*),
                             void  (*dealloc) (void*, size_t));


extern void __cxa_vec_cctor(void*  dest_array, 
                            void*  src_array, 
                            size_t element_count, 
                            size_t element_size, 
                            void  (*constructor) (void*, void*), 
                            void  (*destructor)(void*) );


// 3.3.5.3 Runtime API
extern int __cxa_atexit(void (*f)(void*), void* p, void* d);
extern int __cxa_finalize(void*);


// 3.4 Demangler API
extern char* __cxa_demangle(const char* mangled_name, 
                            char*       output_buffer,
                            size_t*     length, 
                            int*        status);

// Apple additions to support C++ 0x exception_ptr class
// These are primitives to wrap a smart pointer around an exception object
extern void * __cxa_current_primary_exception() throw();
extern void __cxa_rethrow_primary_exception(void* primary_exception);
extern void __cxa_increment_exception_refcount(void* primary_exception) throw();
extern void __cxa_decrement_exception_refcount(void* primary_exception) throw();

// Apple addition to support std::uncaught_exception()
extern bool __cxa_uncaught_exception() throw();

  } // extern "C"
} // namespace __cxxabiv1

namespace abi = __cxxabiv1;

#endif // __cplusplus

#endif // __CXXABI_H 
