/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
#ifndef __XA_NNLIB_ERR_CHK_H__
#define __XA_NNLIB_ERR_CHK_H__

#ifndef NULL
#define NULL (void *)0
#endif /* NULL */

#ifndef DISABLE_ARG_CHK

#define XA_NNLIB_ARG_CHK_PTR(_ptr, _err)                                \
do {                                                                    \
  if((_ptr) == NULL) return (_err);                                     \
} while(0)

#define XA_NNLIB_ARG_CHK_ALIGN(_ptr, _align, _err)                      \
do {                                                                    \
  if(((unsigned int)(_ptr) & ((_align) - 1)) != 0) return (_err);       \
} while(0)

#define XA_NNLIB_ARG_CHK_COND(_cond, _err)                              \
do {                                                                    \
  if((_cond)) return (_err);                                            \
} while(0)

#else /* DISABLE_ARG_CHK */

#define XA_NNLIB_ARG_CHK_PTR(_ptr, _err)
#define XA_NNLIB_ARG_CHK_ALIGN(_ptr, _align, _err)
#define XA_NNLIB_ARG_CHK_COND(_cond, _err)

#endif /* DISABLE_ARG_CHK */

#define XA_NNLIB_CHK_PTR(_ptr, _err)                                    \
do {                                                                    \
  if((_ptr) == NULL) return (_err);                                     \
} while(0)

#define XA_NNLIB_CHK_ALIGN(_ptr, _align, _err)                          \
do {                                                                    \
  if(((unsigned int)(_ptr) & ((_align) - 1)) != 0) return (_err);       \
} while(0)

#define XA_NNLIB_CHK_COND(_cond, _err)                                  \
do {                                                                    \
  if((_cond)) return (_err);                                            \
} while(0)

#endif /* __XA_NNLIB_ERR_CHK_H__ */

