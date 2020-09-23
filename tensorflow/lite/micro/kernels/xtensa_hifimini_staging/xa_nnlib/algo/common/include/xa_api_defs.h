/*******************************************************************************
 * Copyright (c) 2019-2020 Cadence Design Systems, Inc.
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef __XA_API_DEFS_H__
#define __XA_API_DEFS_H__

/*****************************************************************************/
/* Constant hash defines                                                     */
/*****************************************************************************/
/* A constant to let API copy small strings to buffers outside */
#define XA_API_STR_LEN 30
#define XA_APIVERSION_MAJOR 1
#define XA_APIVERSION_MINOR 0

/* last compatible version */
/* sometimes a new API version is just for a bugfix, or a added feature  in */
/* this case it is better to use a newer version even though a library  was */
/* made for an older version, library API can then be upgraded to newer API */
/* version after checking for compatibility or by adding features           */
#define XA_LASTCOMP_APIVERSION_MAJOR 1
#define XA_LASTCOMP_APIVERSION_MINOR 0

#define XA_STR(str) #str
#define XA_MAKE_VERSION_STR(maj, min) XA_STR(maj) "." XA_STR(min)
#define XA_APIVERSION \
  XA_MAKE_VERSION_STR(XA_APIVERSION_MAJOR, XA_APIVERSION_MINOR)

#define XA_LAST_COMP_APIVERSION                     \
  XA_MAKE_VERSION_STR(XA_LASTCOMP_APIVERSION_MAJOR, \
                      XA_LASTCOMP_APIVERSION_MINOR)

#endif /* __XA_API_DEFS_H__ */
