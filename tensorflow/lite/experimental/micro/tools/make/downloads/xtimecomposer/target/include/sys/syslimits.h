/*
 * Copyright (c) 1988, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)syslimits.h	8.1 (Berkeley) 6/2/93
 * $FreeBSD: src/sys/sys/syslimits.h,v 1.10 2001/06/18 20:24:54 wollman Exp $
 */

#ifndef _SYS_SYSLIMITS_H_
#define _SYS_SYSLIMITS_H_

#define	ARG_MAX			65536	/* max bytes for an exec function */
#ifndef CHILD_MAX
#define	CHILD_MAX		   40	/* max simultaneous processes */
#endif
#define	LINK_MAX		32767	/* max file link count */
#define	MAX_CANON		  255	/* max bytes in term canon input line */
#define	MAX_INPUT		  255	/* max bytes in terminal input */
#define	NAME_MAX		  255	/* max bytes in a file name */
#define	NGROUPS_MAX		   16	/* max supplemental group id's */
#ifndef OPEN_MAX
#define	OPEN_MAX		   64	/* max open files per process */
#endif
#define	PATH_MAX		 1024	/* max bytes in pathname */
#define	PIPE_BUF		  512	/* max bytes for atomic pipe writes */
#define	IOV_MAX			 1024	/* max elements in i/o vector */

#define	BC_BASE_MAX		   99	/* max ibase/obase values in bc(1) */
#define	BC_DIM_MAX		 2048	/* max array elements in bc(1) */
#define	BC_SCALE_MAX		   99	/* max scale value in bc(1) */
#define	BC_STRING_MAX		 1000	/* max const string length in bc(1) */
#define	COLL_WEIGHTS_MAX	    0	/* max weights for order keyword */
#define	EXPR_NEST_MAX		   32	/* max expressions nested in expr(1) */
#define	LINE_MAX		 2048	/* max bytes in an input line */
#define	RE_DUP_MAX		  255	/* max RE's in interval notation */

#endif
