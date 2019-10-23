#ifndef	_MACHSTDLIB_H_
#define	_MACHSTDLIB_H_

/* place holder so platforms may add stdlib.h extensions */

#ifndef __XC__
// qsort2() is a non-recursive replacement for the recursive qsort().

// The qsort2() function shall sort an array of __nel objects, the initial
// element of which is pointed to by __base.
// The size of each object, in bytes, is specified by the __width argument.
// If the __nel argument has the value zero, the comparison function pointed to
// by __less shall not be called and no rearrangement shall take place.
// The application must provide an additional temporary element pointed to by __scratch.

// The contents of the array shall be sorted in ascending order according to a
// comparison function.
// The __less argument is a pointer to the comparison function, which is called
// with two arguments that point to the elements being compared.
// The application shall ensure that the function returns zero if the first
// argument is not considered respectively less than than the second.
// If two members compare as equal, their order in the sorted array is unspecified.

// xcore requires arg '__less' to have its fptrgroup attribute set viz:
//    __attribute__((fptrgroup("stdlib_qsort2"))) int myLessThanFunc(void*,void*) {...}
_VOID _EXFUN(qsort2,(_PTR __base, unsigned short __nel, size_t __width,
					 int(*__less)(const _PTR, const _PTR), _PTR __scratch));
#endif

#endif	/* _MACHSTDLIB_H_ */


