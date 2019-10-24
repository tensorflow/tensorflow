/*
 * Copyright (C) XMOS Limited 2013
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

#ifndef _timer_h_
#define _timer_h_

#define XS1_TIMER_MHZ 100U
#define XS1_TIMER_KHZ ((XS1_TIMER_MHZ) * 1000U)
#define XS1_TIMER_HZ ((XS1_TIMER_MHZ) * 1000000U)

#ifndef __ASSEMBLER__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pause the calling task for the specified number of reference clock ticks.
 * \param ticks The number of reference clock ticks to delay execution for.
 */
void delay_ticks(unsigned ticks);

/**
 * Pause the calling task for the specified number of reference clock ticks
 * given as a unsigned long long integer.
 * \param ticks The number of reference clock ticks to delay execution for.
 */
void delay_ticks_longlong(unsigned long long ticks);

/**
 * Pause the calling task for the specified number of seconds.
 * \param delay The number of seconds to delay execution for.
 */
inline void delay_seconds(unsigned int delay) {
  delay_ticks_longlong(XS1_TIMER_MHZ * 1000000 * (unsigned long long)delay);
}

/**
 * Pause the calling task for the specified number of milliseconds.
 * \param delay The number of milliseconds to delay execution for.
 */
inline void delay_milliseconds(unsigned delay) {
  delay_ticks_longlong(XS1_TIMER_MHZ * 1000 * (unsigned long long)delay);
}

/**
 * Pause the calling task for the specified number of microseconds.
 * \param delay The number of microseconds to delay execution for.
 */
inline void delay_microseconds(unsigned delay) {
  delay_ticks_longlong(XS1_TIMER_MHZ * (unsigned long long)delay);
}

#ifdef __cplusplus
} //extern "C"
#endif

#endif // __ASSEMBLER__

#endif // _timer_h_
