/*
 * Copyright (C) XMOS Limited 2008
 * 
 * The copyrights, all other intellectual and industrial property rights are
 * retained by XMOS and/or its licensors.
 *
 * The code is provided "AS IS" without a warranty of any kind. XMOS and its
 * licensors disclaim all other warranties, express or implied, including any
 * implied warranty of merchantability/satisfactory quality, fitness for a
 * particular purpose, or non-infringement except to the extent that these
 * disclaimers are held to be legally invalid under applicable law.
 *
 * Version: Community_15.0.0_eng
 */

#ifndef _xccompat_h_
#define _xccompat_h_

/**
 * \file xccompat.h
 * \brief XC compatibility header for C/C++
 *
 * This header provides type definitions that simplify the task of
 * writing functions that may be called from both C/C++ and XC.
 */

/**
 * Macro that expands to a reference to the specified type when used in
 * an XC source file and to a pointer to the specified type when used in
 * a C/C++ source file.
 */
#ifdef __XC__
#define REFERENCE_PARAM(type, name) type &name
#else
#define REFERENCE_PARAM(type, name) type *name
#endif

/**
 * Macro that expands to a nullable reference to the specified type when used
 * in an XC source file and to a pointer to the specified type when used in
 * a C/C++ source file.
 */
#ifdef __XC__
#define NULLABLE_REFERENCE_PARAM(type, name) type &?name
#else
#define NULLABLE_REFERENCE_PARAM(type, name) type *name
#endif

/**
 * Macro that expands to a nullable resource of the specified type when used
 * in an XC source file and to an integer type capable of holding a resource
 * identifier when used in a C/C++ source file.
 */
#ifdef __XC__
#define NULLABLE_RESOURCE(type, name) type ?name
#else
#define NULLABLE_RESOURCE(type, name) type name
#endif

/**
 * Macro that expands to an array of the specified type when used in an XC
 * source file and to a pointer to the specified type when used in
 * a C/C++ source file.
 */
#ifdef __XC__
#define NULLABLE_ARRAY_OF(type, name) type (&?name)[]
#else
#define NULLABLE_ARRAY_OF(type, name) type *name
#endif

/**
 * Macro that expands to an array of the specified type when used in an XC
 * source file and to a pointer to the specified type when used in
 * a C/C++ source file.
 */
#ifdef __XC__
#define ARRAY_OF_SIZE(type, name, size) type name[size]
#else
#define ARRAY_OF_SIZE(type, name, size) type *name
#endif

/**
 * Macro that expands to an array of the specified type when used in an XC
 * source file and to a pointer to the specified type when used in
 * a C/C++ source file.
 */
#ifdef __XC__
#define NULLABLE_ARRAY_OF_SIZE(type, name, size) type (&?name)[size]
#else
#define NULLABLE_ARRAY_OF_SIZE(type, name, size) type *name
#endif

/**
 * Macro that expands to an client interface with the specified tag when used
 * in an XC source file and to a compatible type when used in a C/C++ source
 * file.
 */
#ifdef __XC__
#define CLIENT_INTERFACE(tag, name) client interface tag name
#else
#define CLIENT_INTERFACE(type, name) unsigned name
#endif

/**
 * Macro that expands to an client interface with the specified tag when used
 * in an XC source file and to a compatible type when used in a C/C++ source
 * file.
 */
#ifdef __XC__
#define SERVER_INTERFACE(tag, name) server interface tag name
#else
#define SERVER_INTERFACE(type, name) unsigned name
#endif

#if !defined(__XC__) || defined(__DOXYGEN__)
/**
 * chanend typedef for use in C/C++ code. This typedef is only supplied
 * if xccompat.h is included from C/C++ code. This enables a XC function
 * prototyped as taking a parameter of type chanend to be called from C and
 * vice versa.
 */
typedef unsigned chanend;
/**
 * timer typedef for use in C/C++ code. This typedef is only supplied
 * if xccompat.h is included from C/C++ code. This enables a XC function
 * prototyped as taking a parameter of type timer to be called from C and
 * vice versa.
 */
typedef unsigned timer;
/**
 * port typedef for use in C/C++ code. This typedef is only supplied
 * if xccompat.h is included from C/C++ code. This enables a XC function
 * prototyped as taking a parameter of type port to be called from C and
 * vice versa.
 */
typedef unsigned port;
/**
 * streaming chanend typedef for use in C/C++ code.
 * \sa chanend
 */
typedef unsigned streaming_chanend_t;

/**
 * in buffered port:1 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned in_buffered_port_1_t;
/**
 * in buffered port:4 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned in_buffered_port_4_t;
/**
 * in buffered port:8 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned in_buffered_port_8_t;
/**
 * in buffered port:16 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned in_buffered_port_16_t;
/**
 * in buffered port:32 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned in_buffered_port_32_t;
/**
 * out buffered port:1 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned out_buffered_port_1_t;
/**
 * out buffered port:4 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned out_buffered_port_4_t;
/**
 * out buffered port:8 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned out_buffered_port_8_t;
/**
 * out buffered port:16 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned out_buffered_port_16_t;
/**
 * out buffered port:32 typedef for use in C/C++ code.
 * \sa port
 */
typedef unsigned out_buffered_port_32_t;

#ifndef _clock_defined
#define _clock_defined
typedef unsigned clock;
#endif /* _clock_defined */

#else

typedef streaming chanend streaming_chanend_t;

typedef in buffered port:1 in_buffered_port_1_t;
typedef in buffered port:4 in_buffered_port_4_t;
typedef in buffered port:8 in_buffered_port_8_t;
typedef in buffered port:16 in_buffered_port_16_t;
typedef in buffered port:32 in_buffered_port_32_t;

typedef out buffered port:1 out_buffered_port_1_t;
typedef out buffered port:4 out_buffered_port_4_t;
typedef out buffered port:8 out_buffered_port_8_t;
typedef out buffered port:16 out_buffered_port_16_t;
typedef out buffered port:32 out_buffered_port_32_t;

#endif

#endif /* _xccompat_h_ */
