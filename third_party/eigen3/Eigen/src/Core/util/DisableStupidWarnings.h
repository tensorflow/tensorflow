#ifndef EIGEN_WARNINGS_DISABLED
#define EIGEN_WARNINGS_DISABLED

#ifdef _MSC_VER
  // 4100 - unreferenced formal parameter (occurred e.g. in aligned_allocator::destroy(pointer p))
  // 4101 - unreferenced local variable
  // 4127 - conditional expression is constant
  // 4181 - qualifier applied to reference type ignored
  // 4211 - nonstandard extension used : redefined extern to static
  // 4244 - 'argument' : conversion from 'type1' to 'type2', possible loss of data
  // 4273 - QtAlignedMalloc, inconsistent DLL linkage
  // 4324 - structure was padded due to declspec(align())
  // 4512 - assignment operator could not be generated
  // 4522 - 'class' : multiple assignment operators specified
  // 4700 - uninitialized local variable 'xyz' used
  // 4717 - 'function' : recursive on all control paths, function will cause runtime stack overflow
  #ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    #pragma warning( push )
  #endif
  #pragma warning( disable : 4100 4101 4127 4181 4211 4244 4273 4324 4512 4522 4700 4717 )
#elif defined __INTEL_COMPILER
  // 2196 - routine is both "inline" and "noinline" ("noinline" assumed)
  //        ICC 12 generates this warning even without any inline keyword, when defining class methods 'inline' i.e. inside of class body
  //        typedef that may be a reference type.
  // 279  - controlling expression is constant
  //        ICC 12 generates this warning on assert(constant_expression_depending_on_template_params) and frankly this is a legitimate use case.
  #ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    #pragma warning push
  #endif
  #pragma warning disable 2196 279
#elif defined __clang__
  // -Wconstant-logical-operand - warning: use of logical && with constant operand; switch to bitwise & or remove constant
  //     this is really a stupid warning as it warns on compile-time expressions involving enums
  #ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    #pragma clang diagnostic push
  #endif
  #pragma clang diagnostic ignored "-Wconstant-logical-operand"
#endif

#endif // not EIGEN_WARNINGS_DISABLED
