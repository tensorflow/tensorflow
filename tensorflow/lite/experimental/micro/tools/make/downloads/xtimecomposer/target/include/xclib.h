/*
 * Copyright (C) XMOS Limited 2008 - 2009
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

#ifndef _xclib_h_
#define _xclib_h_

/** 
 * \file xclib.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns x with the order of the bits reversed. Together with
 * byterev() this function can be used to translate between different
 * ordering conventions such as big-endian and little-endian.
 * \param x
 * \return The bit reversed value.
 * \sa byterev
 */
unsigned bitrev(unsigned x);
#define bitrev(x)         __builtin_bitrev(x)
  
/**
 * Returns x with the order of the bytes reversed. Together with
 * bitrev() this function can be used to translate between different
 * ordering conventions such as big-endian and little-endian.
 * \param x
 * \return The byte reversed value.
 * \sa bitrev
 */
unsigned byterev(unsigned x);
#ifdef __GNUC__
#define byterev(x)        __builtin_bswap32(x)
#else
#define byterev(x)        __builtin_byterev(x)
#endif

/**
 * Returns the number of leading 0-bits in \a x, starting at the most
 * significant bit position.
 * \param x
 * \return The number of leading 0 bits.
 */
int clz(unsigned x);
#define clz(x)            __builtin_clz(x)

#ifdef __cplusplus
} //extern "C" 
#endif

#endif /* _xclib_h_ */
