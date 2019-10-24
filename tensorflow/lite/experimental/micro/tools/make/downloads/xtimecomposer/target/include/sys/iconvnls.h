/*
 * Copyright (c) 2003-2004, Artem B. Bityuckiy.
 * Rights transferred to Franklin Electronic Publishers.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Funtions, macros, etc implimented in iconv library but used by other
 * NLS-related subsystems too.
 */
#ifndef __SYS_ICONVNLS_H__
#define __SYS_ICONVNLS_H__

#include <_ansi.h>
#include <reent.h>
#include <wchar.h>
#include <iconv.h>

/* Iconv data path environment variable name */
#define NLS_ENVVAR_NAME  "NLSPATH"
/* Default NLSPATH value */
#define ICONV_DEFAULT_NLSPATH "/usr/locale"
/* Direction markers */
#define ICONV_NLS_FROM 0
#define ICONV_NLS_TO   1

_VOID
_EXFUN(_iconv_nls_get_state, (iconv_t cd, mbstate_t *ps, int direction));

int
_EXFUN(_iconv_nls_set_state, (iconv_t cd, mbstate_t *ps, int direction));

int
_EXFUN(_iconv_nls_is_stateful, (iconv_t cd, int direction));

int
_EXFUN(_iconv_nls_get_mb_cur_max, (iconv_t cd, int direction));

size_t
_EXFUN(_iconv_nls_conv, (struct _reent *rptr, iconv_t cd,
                        _CONST char **inbuf, size_t *inbytesleft,
                        char **outbuf, size_t *outbytesleft));

_CONST char *
_EXFUN(_iconv_nls_construct_filename, (struct _reent *rptr, _CONST char *file,
                                      _CONST char *dir, _CONST char *ext));


int
_EXFUN(_iconv_nls_open, (struct _reent *rptr, _CONST char *encoding,
                        iconv_t *towc, iconv_t *fromwc, int flag));

char *
_EXFUN(_iconv_resolve_encoding_name, (struct _reent *rptr, _CONST char *ca));

#endif /* __SYS_ICONVNLS_H__ */

