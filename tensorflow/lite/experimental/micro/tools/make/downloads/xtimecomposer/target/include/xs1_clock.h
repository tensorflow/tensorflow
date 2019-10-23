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

#ifndef _xs1_clock_h_
#define _xs1_clock_h_

#ifndef __ASSEMBLER__
  
#ifndef _clock_defined
#define _clock_defined
#ifdef __XC__
/**
 * Clock resource type. Clocks are declared as global variables and are
 * initialized with the resource identifier of a clock block. When in a running
 * state a clock provides rising and falling edges to ports configured using
 * that clock.
 * \sa start_clock
 * \sa stop_clock
 * \sa configure_clock_src
 * \sa configure_clock_rate
 * \sa configure_clock_ref
 * \sa configure_clock_rate_at_least
 * \sa configure_clock_rate_at_most
 */
#define clock __clock_t
#else
typedef unsigned clock;
#endif
#endif /* _clock_defined */

#endif // __ASSEMBLER__

#endif // _xs1_clock_h_
