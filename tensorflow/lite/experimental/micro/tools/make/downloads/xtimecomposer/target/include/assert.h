/*
	assert.h
*/

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#include "_ansi.h"

#undef assert

#ifdef NDEBUG           /* required by ANSI standard */
# define assert(__e) ((void)0)
#else

# ifdef __XC__
#  define assert(__e) do { if (!(__e)) { __assert (__FILE__, __LINE__, #__e); } } while(0)
# else
#  define assert(__e) ((__e) ? (void)0 : __assert_func (__FILE__, __LINE__, \
							__ASSERT_FUNC, #__e))

#  ifndef __ASSERT_FUNC
  /* Use g++'s demangled names in C++.  */
#   if defined __cplusplus && defined __GNUC__
#    define __ASSERT_FUNC __PRETTY_FUNCTION__

  /* C99 requires the use of __func__.  */
#   elif __STDC_VERSION__ >= 199901L
#    define __ASSERT_FUNC __func__

  /* Older versions of gcc don't have __func__ but can use __FUNCTION__.  */
#   elif __GNUC__ >= 2
#    define __ASSERT_FUNC __FUNCTION__

  /* failed to detect __func__ support.  */
#   else
#    define __ASSERT_FUNC ((char *) 0)
#   endif
#  endif /* !__ASSERT_FUNC */
# endif /* !NDEBUG */
#endif /* !__XC__ */

void _EXFUN(__assert, (const char *__file, int, const char *__message)
	    _ATTRIBUTE ((__noreturn__)));
void _EXFUN(__assert_func, (const char *__file, int, const char *__function, const char *__message)
	    _ATTRIBUTE ((__noreturn__)));

#if defined(__cplusplus) || defined(__XC__)
}
#endif
