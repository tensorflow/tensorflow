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

#ifndef _safestring_h_
#define _safestring_h_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file safestring.h
 * \brief XC standard string routines.
 *
 * The safestring library provides safe versions of the string functions 
 * found in string.h of the C standard library. All functions are callable
 * from XC. When called from XC any attempt to perform an out of bounds
 * array access will cause an exception to be raised.
 */

/**
 * Copies a string (including the terminating null character) to an array.
 * \param s1 The array to copy to.
 * \param s2 The string to copy.
 * \sa safestrncpy
 */
void safestrcpy(char s1[], const char s2[]);

/**
 * Copies no more than \a n characters of the string \a s1 to the array \a s2.
 * If the length of \a s2 is less than \a n then null characters will be
 * appended to the copied characters until \a n bytes are written.
 * \param s1 The array to copy to.
 * \param s2 The string to copy.
 * \param n The number of characters to copy.
 * \sa safestrcpy
 */
void safestrncpy(char s1[], const char s2[], unsigned n);

/**
 * Appends a copy of a string (including the terminating null character)
 * to the end of another string.
 * \param s1 The string to append to.
 * \param s2 The string to append.
 * \sa safestrncat
 */
void safestrcat(char s1[], const char s2[]);

/**
 * Appends no more than \a n characters of the string \a s2 to the string
 * \a s1. The null characters at the end of \a s1 is overwritten by the
 * first character of \a s2. A terminating null character is always appended
 * to the result.
 * \param s1 The string to append to.
 * \param s2 The string to append.
 * \param n The number of characters to append.
 * \sa safestrcat
 */
void safestrncat(char s1[], const char s2[], unsigned n);

/**
 * Compares two strings. If the strings are equal 0 is returned. If the strings
 * are not equal a non-zero value is returned, the sign of which is determined
 * by the sign of the difference between the first pair of characters which
 * differ in the strings being compared.
 * \param s1 The first string to compare.
 * \param s2 The second string to compare.
 * \return A integer greater than, equal to, or less than 0, if \a s1 is
 *         respectively greater than, equal to, or less than \a s2.
 * \sa safestrncmp
 */
int safestrcmp(const char s1[], const char s2[]);

/**
 * Compares up to the first \a n character of two strings. If the strings are
 * equal up to the first \a n characters, 0 is returned. Otherwise a non-zero
 * value is returned, the sign of which is determined by the sign of the
 * difference between the first pair of characters which differ.
 * \param s1 The first string to compare.
 * \param s2 The second string to compare.
 * \param n The maximum number of characters to compare.
 * \return A integer greater than, equal to, or less than 0, if \a s1 is
 *         respectively greater than, equal to, or less than \a s2.
 * \sa safestrcmp
 */
int safestrncmp(const char s1[], const char s2[], unsigned n);

/**
 * Returns the length of a string. The length is given by the number of
 * characters in the string not including the terminating null character.
 * \param s The string.
 * \return The length of the string.
 */
int safestrlen(const char s[]);

/**
 * Returns the index of the first occurrence of \a c
 * (converted to a <tt>char</tt>) in \a s. If \a c does not occur in
 * \a s, -1 is returned. The terminating null character is considered to be part of
 * \a s.
 * \param s The string to scan.
 * \param c The character to scan for.
 * \return The index of \a c.
 * \sa safestrrchr
 */
int safestrchr(const char s[], int c);

/**
 * Returns the index of the last occurrence of \a c
 * (converted to a <tt>char</tt>) in \a s, or -1 if \a c does
 * not occur in \a s. The terminating null character is considered to be part of
 * \a s.
 * \param s The string to scan.
 * \param c The character to scan for.
 * \return The index of \a c.
 * \sa safestrchr
 */
int safestrrchr(const char s[], int c);

/**
 * Returns the length of the longest initial segment of \a s1 which consists
 * entirely of characters from \a s2.
 * \param s1 The string to scan.
 * \param s2 The string containing characters to scan for.
 * \return The length of the initial segment.
 * \sa safestrcspn
 */
unsigned safestrspn(const char s1[], const char s2[]);

/**
 * Returns the length of the longest initial segment of \a s1 which consists
 * entirely of characters not from \a s2.
 * \param s1 The string to scan.
 * \param s2 The string containing characters to scan for.
 * \return The length of the initial segment.
 * \sa safestrspn
 */
unsigned safestrcspn(const char s1[], const char s2[]);

/**
 * Returns the index of the first occurrence in \a s1 of any character in
 * \a s2. If no character in \a s2 occurs in \a s1, -1 is returned.
 * \param s1 The string to scan.
 * \param s2 The string containing characters to scan for.
 * \return The index of first matching character.
 */
int safestrpbrk(const char s1[], const char s2[]);

/**
 * Returns the index of the first occurrence of \a s1 as a sequence of
 * characters (excluding the terminating null character) in \a s2.
 * If \a s1 is not contained in \a s2, -1 is returned. If \a s2
 * is a zero length string then 0 is returned.
 * \param s1 The string to scan.
 * \param s2 The string to scan for.
 * \return The index of first matching subsequence.
 */
int safestrstr(const char s1[], const char s2[]);

/**
 * Copies \a length bytes from the array \a src to the array \a dst.
 * \param dst The destination array.
 * \param src The source array.
 * \param length The number of bytes to copy.
 * \sa safememset
 * \sa safememcmp
 * \sa safememchr
 */
#if defined(__XC__)
void safememcpy(unsigned char dst[length], const unsigned char src[length],
                unsigned length);
#else
void safememcpy(unsigned char *dst, const unsigned char *src, unsigned length);
#endif

/**
 * Copies \a length bytes from offset \a src of array \a data to offset \a dst
 * of array \a data. If the source and destination areas overlap then copying
 * takes place as if the bytes are first copied from the source into a temporary
 * array and then copied to the destination.
 * \param data The array to move data in.
 * \param dst The offset of the destination area.
 * \param src The offset of the source area.
 * \param length The number of bytes to copy.
 */
void safememmove(unsigned char data[], unsigned dst, unsigned src,
                 unsigned length);

/**
 * Copies \a value (converted to an <tt>unsigned char</tt>) to each of the
 * first \a length bytes of the array \a dst.
 * \param dst The destination array.
 * \param value The value to copy.
 * \param length The number of bytes to copy.
 * \sa safememcpy
 * \sa safememcmp
 * \sa safememchr
 */
#if defined(__XC__)
void safememset(unsigned char dst[length], int value, unsigned length);
#else
void safememset(unsigned char *dst, int value, unsigned length);
#endif

/**
 * Compares the first \a length bytes of the arrays \a s1 and \a s2.
 * If there is no difference 0 is returned, otherwise a non-zero value is
 * returned, the sign of which is determined by the sign of the difference
 * between the first pair of bytes which differ.
 * \param s1 The first array.
 * \param s2 The second array.
 * \param length The number of bytes to compare.
 * \return A integer greater than, equal to, or less than 0, if the first
           \a length bytes of \a s1 are respectively greater than, equal to,
           or less than the first \a length bytes of \a s2.
 * \sa safememcpy
 * \sa safememset
 * \sa safememchr
 */
int safememcmp(const unsigned char s1[], const unsigned char s2[],
               unsigned length);

/**
 * Returns the index of the first occurrence of \a c
 * (converted to an <tt>unsigned char</tt>) in the first \a length bytes of
 * \a s. If \a c does not occur in \a s, -1 is returned.
 * \param s The array to scan.
 * \param c The character to scan for.
 * \param length The number of bytes to scan.
 * \return The index of \a c.
 * \sa safememcpy
 * \sa safememset
 * \sa safememcmp
 */
int safememchr(const unsigned char s[], int c, unsigned length);

#ifdef __cplusplus
} //extern "C" 
#endif

#endif // _xcstring_h_
