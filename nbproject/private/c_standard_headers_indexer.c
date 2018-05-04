/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * Copyright (c) 2016 Oracle and/or its affiliates. All rights reserved.
 *
 * Oracle and Java are registered trademarks of Oracle and/or its affiliates.
 * Other names may be trademarks of their respective owners.
 *
 * The contents of this file are subject to the terms of either the GNU
 * General Public License Version 2 only ("GPL") or the Common
 * Development and Distribution License("CDDL") (collectively, the
 * "License"). You may not use this file except in compliance with the
 * License. You can obtain a copy of the License at
 * http://www.netbeans.org/cddl-gplv2.html
 * or nbbuild/licenses/CDDL-GPL-2-CP. See the License for the
 * specific language governing permissions and limitations under the
 * License.  When distributing the software, include this License Header
 * Notice in each file and include the License file at
 * nbbuild/licenses/CDDL-GPL-2-CP.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the GPL Version 2 section of the License file that
 * accompanied this code. If applicable, add the following below the
 * License Header, with the fields enclosed by brackets [] replaced by
 * your own identifying information:
 * "Portions Copyrighted [year] [name of copyright owner]"
 *
 * If you wish your version of this file to be governed by only the CDDL
 * or only the GPL Version 2, indicate your decision by adding
 * "[Contributor] elects to include this software in this distribution
 * under the [CDDL or GPL Version 2] license." If you do not indicate a
 * single choice of license, a recipient has the option to distribute
 * your version of this file under either the CDDL, the GPL Version 2 or
 * to extend the choice of license to its licensees as provided above.
 * However, if you add GPL Version 2 code and therefore, elected the GPL
 * Version 2 license, then the option applies only if the new code is
 * made subject to such option by the copyright holder.
 *
 * Contributor(s):
 */

// List of standard headers was taken in http://en.cppreference.com/w/c/header

#include <assert.h> 	 // Conditionally compiled macro that compares its argument to zero
#include <ctype.h> 	 // Functions to determine the type contained in character data
#include <errno.h> 	 // Macros reporting error conditions
#include <float.h> 	 // Limits of float types
#include <limits.h> 	 // Sizes of basic types
#include <locale.h> 	 // Localization utilities
#include <math.h> 	 // Common mathematics functions
#include <setjmp.h> 	 // Nonlocal jumps
#include <signal.h> 	 // Signal handling
#include <stdarg.h> 	 // Variable arguments
#include <stddef.h> 	 // Common macro definitions
#include <stdio.h> 	 // Input/output
#include <string.h> 	 // String handling
#include <stdlib.h> 	 // General utilities: memory management, program utilities, string conversions, random numbers
#include <time.h> 	 // Time/date utilities
#include <iso646.h>      // (since C95) Alternative operator spellings
#include <wchar.h>       // (since C95) Extended multibyte and wide character utilities
#include <wctype.h>      // (since C95) Wide character classification and mapping utilities
#ifdef _STDC_C99
#include <complex.h>     // (since C99) Complex number arithmetic
#include <fenv.h>        // (since C99) Floating-point environment
#include <inttypes.h>    // (since C99) Format conversion of integer types
#include <stdbool.h>     // (since C99) Boolean type
#include <stdint.h>      // (since C99) Fixed-width integer types
#include <tgmath.h>      // (since C99) Type-generic math (macros wrapping math.h and complex.h)
#endif
#ifdef _STDC_C11
#include <stdalign.h>    // (since C11) alignas and alignof convenience macros
#include <stdatomic.h>   // (since C11) Atomic types
#include <stdnoreturn.h> // (since C11) noreturn convenience macros
#include <threads.h>     // (since C11) Thread library
#include <uchar.h>       // (since C11) UTF-16 and UTF-32 character utilities
#endif
