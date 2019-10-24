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
#ifndef __hw_timer_h__
#define __hw_timer_h__

/** Hardware timer
 *
 * The hwtimer_t type can be used just like the timer type. It gives a unique
 * hardware timer to use (as opposed to the default timer in XC which is
 * allocated based on a shared hardware timer per logical core).
 *
 */
#ifdef __XC__
typedef [[hwtimer]] timer hwtimer_t;
#else
typedef unsigned int hwtimer_t;
#endif

#endif // __hw_timer_h__
