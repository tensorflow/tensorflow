/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#ifndef __XA_NNLIB_DEBUG_H__
#define __XA_NNLIB_DEBUG_H__

/* Enable for debug prints in this code up to MAX_DEBUG_LEVEL */
//#define API_DEBUG

#if defined(API_DEBUG)

#include <stdio.h>

/* ==================================================================== */
#define PRINT_HEX_STR(str) \
  printf("[%s]\r\n", str); \

#define DISP_DEC_UI8(var)  printf("%d ",            (unsigned char) var)
#define DISP_DEC_I8(var)   printf("%d ",            (char) var)
#define DISP_DEC_I16(var)  printf("%d ",            (short) var)
#define DISP_DEC_I32(var)  printf("%ld ",           (long int) var)
#define DISP_DEC_I64(var)  printf("%lld ",          (long long int) var)
#define DISP_DEC_F32(var)  printf("%f ",            (float) var)

#define PRINT_DEC_UI8(var) printf("[%s] %d \r\n",   #var, (unsigned char) var)
#define PRINT_DEC_I8(var)  printf("[%s] %d \r\n",   #var, (char) var)
#define PRINT_DEC_UI16(var) printf("[%s] %d \r\n",   #var, (unsigned short) var)
#define PRINT_DEC_I16(var) printf("[%s] %d \r\n",   #var, (short) var)
#define PRINT_DEC_I32(var) printf("[%s] %ld \r\n",  #var, (long int) var)
#define PRINT_DEC_I64(var) printf("[%s] %lld \r\n", #var, (long long int) var)

#define DISP_HEX_UI8(var) printf("0x%02x ", (0xFF &    (unsigned int) var))
#define DISP_HEX_I8(var)  printf("0x%02x ", (0xFF &    (unsigned int) var))
#define DISP_HEX_UI16(var) printf("0x%04x ", (0xFFFF &    (unsigned int) var))
#define DISP_HEX_I16(var) printf("0x%04x ", (0xFFFF &    (unsigned int) var))
#define DISP_HEX_I32(var) printf("0x%08x ", (0xFFFFFFFF &    (unsigned int) var))
#define DISP_HEX_I64(var) printf("0x%016llx ", ((long long unsigned int) var))
// #define DISP_HEX_I64(var) printf("0x%016llx ", (0xFFFFFFFFFFFFFFFF & (long long unsigned int) var))

#define DISP_HEX_ADDR(addr) printf("%p ", (addr))

typedef union _float_in_hex_t {
 FLOAT32 f_val;
 UWORD32 u_val;
} float_in_hex_t;

#define DISP_HEX_F32(var) { \
 float_in_hex_t _tmp; \
 _tmp.f_val = var; \
 printf("0x%x ", (unsigned int) _tmp.u_val); \
}

#define PRINT_HEX_UI8(var) printf("[%s] 0x%02x \r\n",    #var, (0xFF & (unsigned char) var))
#define PRINT_HEX_I8(var)  printf("[%s] 0x%02x \r\n",    #var, (0xFF & ( char) var))
#define PRINT_HEX_UI16(var) printf("[%s] 0x%04x \r\n",    #var, (0xFFFF & ( unsigned short) var))
#define PRINT_HEX_I16(var) printf("[%s] 0x%04x \r\n",    #var, (0xFFFF & ( short) var))
#define PRINT_HEX_I32(var) printf("[%s] 0x%08x \r\n",    #var, (0xFFFFFFFF & ( long int) var))
#define PRINT_HEX_I64(var) printf("[%s] 0x%016llx \r\n", #var, (0xFFFFFFFFFFFFFFFF & ( long long int) var))
/* ==================================================================== */


__attribute__((unused))
static void print_mat_u8b(UWORD8 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_UI8(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_UI8(mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_8b(WORD8 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I8(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I8(mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_16b(WORD16 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I16(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I16(mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_32b(WORD32 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s]===========\r\n", __func__, mat_name);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I32(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I32(mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_64b(WORD64 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  //printf("\r\n===========[%s][%s]===========\r\n", __func__, mat_name);
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I64(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I64(mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_f32b(FLOAT32 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s]===========\r\n", __func__, mat_name);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_F32(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_F32(mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

#define DISP_HEX_16x4(var) \
  print_mat_16b((WORD16 *)&var, 1, 4, 1, 4, #var); \

/* Max level up to which debug prints will be enabled. */
#define MAX_DEBUG_LEVEL 3

#if (MAX_DEBUG_LEVEL < 0)
#error "Incorrect value selected. Chose MAX_DEBUG_LEVEL >= 0"
#endif /* (MAX_DEBUG_LEVEL < 1) */

/* Debug prints using printf */
// #define SHOW_FILE_LINE_FUNC_INFO
#if defined(SHOW_FILE_LINE_FUNC_INFO)
#define FORMAT_FOR_FILE_LINE_FUNC_INFO "%s:%d:%s"
#define VALUES_FOR_FILE_LINE_FUNC_INFO __FILE__, __LINE__, __func__
#else
#define FORMAT_FOR_FILE_LINE_FUNC_INFO "(%s)"
#define VALUES_FOR_FILE_LINE_FUNC_INFO "Empty"
#endif /* SHOW_FILE_LINE_FUNC_INFO */

#define DP_1(fmt, ...) if(1 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: " FORMAT_FOR_FILE_LINE_FUNC_INFO fmt, 1, VALUES_FOR_FILE_LINE_FUNC_INFO, ## __VA_ARGS__)
#define DP_2(fmt, ...) if(2 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: " FORMAT_FOR_FILE_LINE_FUNC_INFO fmt, 2, VALUES_FOR_FILE_LINE_FUNC_INFO, ## __VA_ARGS__)
#define DP_3(fmt, ...) if(3 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: " FORMAT_FOR_FILE_LINE_FUNC_INFO fmt, 3, VALUES_FOR_FILE_LINE_FUNC_INFO, ## __VA_ARGS__)
#define DP_4(fmt, ...) if(4 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: " FORMAT_FOR_FILE_LINE_FUNC_INFO fmt, 4, VALUES_FOR_FILE_LINE_FUNC_INFO, ## __VA_ARGS__)
#define DP_5(fmt, ...) if(5 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: " FORMAT_FOR_FILE_LINE_FUNC_INFO fmt, 5, VALUES_FOR_FILE_LINE_FUNC_INFO, ## __VA_ARGS__)
#if 0
#define DP_1(fmt, ...) if(1 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: %s:%d:%s(): " fmt, 1, __FILE__, __LINE__, __func__, ## __VA_ARGS__)
#define DP_2(fmt, ...) if(2 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: %s:%d:%s(): " fmt, 2, __FILE__, __LINE__, __func__, ## __VA_ARGS__)
#define DP_3(fmt, ...) if(3 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: %s:%d:%s(): " fmt, 3, __FILE__, __LINE__, __func__, ## __VA_ARGS__)
#define DP_4(fmt, ...) if(4 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: %s:%d:%s(): " fmt, 4, __FILE__, __LINE__, __func__, ## __VA_ARGS__)
#define DP_5(fmt, ...) if(5 <= MAX_DEBUG_LEVEL) printf("\r\n\n[D%d]: %s:%d:%s(): " fmt, 5, __FILE__, __LINE__, __func__, ## __VA_ARGS__)
#endif /* 0 */

#define DBG_SELF_INFO DP_1("[Self]")

static inline _Bool is_aligned(const void *restrict pointer, size_t byte_count)
  {
    return (unsigned int)pointer % byte_count == 0;
  }

#define PTR_ALIGNMENT_STATUS(_ptr, _byte_count) \
  { \
    DP_1("\r\n[PTR]:" #_ptr "\r\n\t[SELF_ADDR]:[%p]\r\n\t[BOUNDARY]:[%d]\r\n\t[PTR_ALIGNMENT_STATUS]:[%s]\r\n\t[VALUE]:[%p]", (void *) &_ptr, _byte_count, \
      ((!_ptr) ? "NULL" : (is_aligned((const void *restrict) _ptr, (size_t) _byte_count)) ? "Aligned" : "Unaligned"), \
      ((!_ptr) ? "0x0000" : _ptr)); \
  }

#define PRINT_HEX_STATEMENT_FOR_NULL_POINTER_WITH_RETURN {DP_1("[RETURN]"); printf("Null pointers received. Returning back.!\r\n"); return;}

#define RETURN_IF_ANY_NULL_FOUND_2(_ptr_1, _ptr_2) \
  if(!PROCEED_IF_NOT_NULL_2(_ptr_1, _ptr_2)) PRINT_HEX_STATEMENT_FOR_NULL_POINTER_WITH_RETURN

#define RETURN_IF_ANY_NULL_FOUND_3(_ptr_1, _ptr_2, _ptr_3) \
  if(!PROCEED_IF_NOT_NULL_3(_ptr_1, _ptr_2, _ptr_3)) PRINT_HEX_STATEMENT_FOR_NULL_POINTER_WITH_RETURN

#define RETURN_IF_ALL_PAIRS_FOUND_NULL_4(_ptr_1, _ptr_2, _ptr_3, _ptr_4) \
  if((!PROCEED_IF_NOT_NULL_2(_ptr_1, _ptr_2)) | (!PROCEED_IF_NOT_NULL_2(_ptr_3, _ptr_4))) PRINT_HEX_STATEMENT_FOR_NULL_POINTER_WITH_RETURN

#define DBG_u8xu8_u8(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_u8xu8_u8(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_8x8_8(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_8x8_8(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_8x8_16(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_8x8_16(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_8x8_32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_8x8_32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_8x8_8_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,bias_precision,p_scratch) \
  dbg_8x8_8_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,bias_precision,p_scratch);

#define DBG_8x16_16(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_8x16_16(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_8x16_32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_8x16_32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_8x16_64(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_8x16_64(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_8x16_16_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,bias_precision,p_scratch) \
  dbg_8x16_16_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,bias_precision,p_scratch);

#define DBG_16x16_16(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_16x16_16(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_16x16_32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_16x16_32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_16x16_64(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_16x16_64(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_16x16_16_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,bias_precision,p_scratch) \
  dbg_16x16_16_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,bias_precision,p_scratch);

#define DBG_f32xf32_f32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2) \
  dbg_f32xf32_f32(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2);

#define DBG_f32xf32_f32_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,p_scratch) \
  dbg_f32xf32_f32_activation(p_out,p_mat1,p_mat2,p_vec1,p_vec2,p_bias,rows,cols1,cols2,row_stride1,row_stride2,p_scratch);

__attribute__((unused))
static void dbg_u8xu8_u8(
    UWORD8 *p_out,
    UWORD8 *p_mat1,
    UWORD8 *p_mat2,
    UWORD8 *p_vec1,
    UWORD8 *p_vec2,
    UWORD8 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_u8b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_u8b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_u8b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_u8b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_u8b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_u8b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x8_8(
    WORD8 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD8 *p_vec1,
    WORD8 *p_vec2,
    WORD8 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_8b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_8b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_8b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_8b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_8b(p_out, rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x8_16(
    WORD16 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD8 *p_vec1,
    WORD8 *p_vec2,
    WORD8 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_8b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_8b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_8b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_8b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_16b(p_out, rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x8_32(
    WORD32 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD8 *p_vec1,
    WORD8 *p_vec2,
    WORD8 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_8b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_8b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_8b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_8b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_32b(p_out, rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x8_8_activation(
    WORD8 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD8 *p_vec1,
    WORD8 *p_vec2,
    VOID   *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    WORD32 bias_precision,
    VOID   *p_scratch)
{
  print_mat_8b(p_mat1,     rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2,     rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_8b(p_vec1,     row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_8b(p_vec2,     row_stride2, 1,           cols2, 1,     "p_vec2");

  if(64 == bias_precision)
  {
    print_mat_8b(p_bias,   rows,        1,           rows,  1,     "p_bias");
  }
  else if(32 == bias_precision)
  {
    print_mat_32b(p_bias,  rows,        1,           rows,  1,     "p_bias");
  }

  print_mat_32b(p_scratch, rows,        1,           rows,  1,     "p_scratch");
  print_mat_8b(p_out,      rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x16_16(
    WORD16 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_8b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_16b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_16b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x16_32(
    WORD32 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_8b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_16b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_32b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x16_64(
    WORD64 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_8b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_16b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_64b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_8x16_16_activation(
    WORD16 *p_out,
    WORD8 *p_mat1,
    WORD8 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    VOID   *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    WORD32 bias_precision,
    VOID   *p_scratch)
{
  print_mat_8b(p_mat1,    rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_8b(p_mat2,    rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1,    row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2,    row_stride2, 1,           cols2, 1,     "p_vec2");

  if(16 == bias_precision)
  {
    print_mat_16b(p_bias,  rows,        1,           rows,  1,     "p_bias");
  }
  else if(64 == bias_precision)
  {
    print_mat_64b(p_bias,  rows,        1,           rows,  1,     "p_bias");
  }

  print_mat_32b(p_scratch, rows,        1,           rows,  1,     "p_scratch");
  print_mat_16b(p_out,     rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_16x16_16(
    WORD16 *p_out,
    WORD16 *p_mat1,
    WORD16 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_16b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_16b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_16b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_16b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_16x16_32(
    WORD32 *p_out,
    WORD16 *p_mat1,
    WORD16 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_16b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_16b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_16b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_32b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_16x16_64(
    WORD64 *p_out,
    WORD16 *p_mat1,
    WORD16 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    WORD16 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_16b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_16b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_16b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_64b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_16x16_16_activation(
    WORD16 *p_out,
    WORD16 *p_mat1,
    WORD16 *p_mat2,
    WORD16 *p_vec1,
    WORD16 *p_vec2,
    VOID   *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    WORD32 bias_precision,
    VOID   *p_scratch)
{
  print_mat_16b(p_mat1,    rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_16b(p_mat2,    rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_16b(p_vec1,    row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_16b(p_vec2,    row_stride2, 1,           cols2, 1,     "p_vec2");

  if(16 == bias_precision)
  {
    print_mat_16b(p_bias,  rows,        1,           rows,  1,     "p_bias");
  }
  else if(64 == bias_precision)
  {
    print_mat_64b(p_bias,  rows,        1,           rows,  1,     "p_bias");
  }

  print_mat_32b(p_scratch, rows,        1,           rows,  1,     "p_scratch");
  print_mat_16b(p_out,     rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_f32xf32_f32(
    FLOAT32 *p_out,
    FLOAT32 *p_mat1,
    FLOAT32 *p_mat2,
    FLOAT32 *p_vec1,
    FLOAT32 *p_vec2,
    FLOAT32 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  print_mat_f32b(p_mat1, rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_f32b(p_mat2, rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_f32b(p_vec1, row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_f32b(p_vec2, row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_f32b(p_bias, rows,        1,           rows,  1,     "p_bias");
  print_mat_f32b(p_out,  rows,        1,           rows,  1,     "p_out");
}

__attribute__((unused))
static void dbg_f32xf32_f32_activation(
    FLOAT32 *p_out,
    FLOAT32 *p_mat1,
    FLOAT32 *p_mat2,
    FLOAT32 *p_vec1,
    FLOAT32 *p_vec2,
    FLOAT32 *p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32 *p_scratch)
{
  print_mat_f32b(p_mat1,    rows,        row_stride1, rows,  cols1, "p_mat1");
  print_mat_f32b(p_mat2,    rows,        row_stride2, rows,  cols2, "p_mat2");
  print_mat_f32b(p_vec1,    row_stride1, 1,           cols1, 1,     "p_vec1");
  print_mat_f32b(p_vec2,    row_stride2, 1,           cols2, 1,     "p_vec2");
  print_mat_f32b(p_bias,    rows,        1,           rows,  1,     "p_bias");
  print_mat_f32b(p_out,     rows,        1,           rows,  1,     "p_out");
  print_mat_f32b(p_scratch, rows,        1,           rows,  1,     "p_scratch");
}

/* Basic formula to calculate result dimension */
#define RESULT_DIM_OF_CONV(_padding, _inp_dim, _kernel_dim, _stride) \
  ((WORD32)((_padding + _inp_dim - _kernel_dim) / (_stride)) + 1)

__attribute__((unused))
static void print_padded_inp_mat_16b
  (pWORD16 p_inp
   ,WORD32 kernel_height
   ,WORD32 kernel_width
   ,WORD32 input_height
   ,WORD32 input_width
   ,WORD32 output_height
   ,WORD32 output_width
   ,WORD32 y_stride
   ,WORD32 x_stride
   ,WORD32 y_padding
   ,WORD32 x_padding
   ,WORD32 input_channels
  )
{
  int i, j;
  int itr_1;
  pWORD16 _pWORD16_inp;
  char name[60] = {0};

  DP_1("==> Imagine how padded input data would look like");

  WORD32 extra_y_padding;
  WORD32 extra_x_padding;

  extra_y_padding =
    y_stride *
      (output_height - RESULT_DIM_OF_CONV(y_padding, input_height, kernel_height, y_stride));

  extra_x_padding =
    x_stride *
      (output_width - RESULT_DIM_OF_CONV(x_padding, input_width, kernel_width, x_stride));

  WORD32 input_height_pad = (y_padding + input_height + extra_y_padding);
  WORD32 input_width_pad  = (x_padding + input_width + extra_x_padding);

  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    _pWORD16_inp = (p_inp + (itr_1 * input_height * input_width));

    sprintf(name, "padded_p_inp_for_inp-ch_%d", itr_1);

    DP_1("\r\n===========[%s][%s][%p]====|%dx%d|=======\r\n", __func__, name, (void *) _pWORD16_inp, input_height_pad, input_width_pad);

    for (i = 0; i < y_padding; i++)
    {
      printf("\r\n|Row = %d|", i);
      for (j = 0; j < input_width_pad; j++)
      {
        DISP_DEC_I16(0);
        printf("|");
      }
      printf("\r\n");
    }
    for (i = 0; i < input_height; i++)
    {
      printf("\r\n|Row = %d|", (i + y_padding));
      for (j = 0; j < x_padding; j++)
      {
        DISP_DEC_I16(0);
        printf("|");
      }
      for (j = 0; j < input_width; j++)
      {
        DISP_DEC_I16(_pWORD16_inp[(i * input_width + j)]);
        printf("|");
      }
      for (j = 0; j < extra_x_padding; j++)
      {
        DISP_DEC_I16(0);
        printf("|");
      }
      printf("\r\n");
    }
    for (i = 0; i < extra_y_padding; i++)
    {
      printf("\r\n|Row = %d|", (i + y_padding + input_height));
      for (j = 0; j < input_width_pad; j++)
      {
        DISP_DEC_I16(0);
        printf("|");
      }
      printf("\r\n");
    }
  }
}

__attribute__((unused))
static void print_mat_8b_adv(WORD8 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I8(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I8(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_8b_cir(WORD8 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  /* Please note that:
   * Updates in pointer will happen after each row print. Thus it is expected that src pointer is continuous for atleast one row. */
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I8(mat[j]);
    }

    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I8(mat[j]);
    }

    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[j]);
    }
    printf("|\r\n");

    /* Increment current pointer in terms of 'bytes' which is the size of one row */
    AE_ADDCIRC16X4_XC((ae_int16x4 *) mat, sizeof(WORD8)*cols);
  }
}

__attribute__((unused))
static void print_mat_64b_adv(WORD64 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I64(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I64(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_32b_adv(WORD32 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I32(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I32(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_16b_adv(WORD16 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I16(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I16(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_16b_cir(WORD16 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  /* Please note that:
   * Updates in pointer will happen after each row print. Thus it is expected that src pointer is continuous for atleast one row. */
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_I16(mat[j]);
    }

    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_I16(mat[j]);
    }

    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[j]);
    }
    printf("|\r\n");

    /* Increment current pointer in terms of 'bytes' which is the size of one row */
    AE_ADDCIRC16X4_XC((ae_int16x4 *) mat, sizeof(WORD16)*cols);
  }
}

__attribute__((unused))
static void print_mat_f32b_adv(FLOAT32 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_F32(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_F32(mat[i*cols + j]);
    }
    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[i*cols + j]);
    }
    printf("|\r\n");
  }
}

__attribute__((unused))
static void print_mat_f32b_cir(FLOAT32 *mat, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name){
  /* Please note that:
   * Updates in pointer will happen after each row print. Thus it is expected that src pointer is continuous for atleast one row. */
  int i,j;
  printf("\r\n===========[%s][%s][%p]====|%dx%d => %dx%d|=======\r\n", __func__, mat_name, (void *) mat, rows, cols, rows_to_show, cols_to_show);
  for (i = 0; i < (rows_to_show < rows ? rows_to_show : rows); i++){
    printf("\r\n|Row = %d|", i);
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_DEC_F32(mat[j]);
    }

    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_F32(mat[j]);
    }

    printf("|");
    for (j = 0; j < (cols_to_show < cols ? cols_to_show : cols); j++){
      DISP_HEX_ADDR(&mat[j]);
    }
    printf("|\r\n");

    /* Increment current pointer in terms of 'bytes' which is the size of one row */
    AE_ADDCIRC16X4_XC((ae_int16x4 *) mat, sizeof(FLOAT32)*cols);
  }
}

__attribute__((unused))
static void dbg_convds_8x16_16
  (pWORD16 p_out
  ,pWORD8  p_kernel
  ,pWORD16 p_inp
  ,pWORD16 p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  out_height
  ,WORD32  out_width
  )
{
  int itr_1;
  int itr_2;
  int tmp;
  char name[60] = {0};

  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_kernel_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_8b_adv((p_kernel + (tmp * kernel_height * kernel_width)), kernel_height, kernel_width, kernel_height, kernel_width, name);
    }
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    sprintf(name, "p_inp_for_inp-ch_%d", itr_1);
    print_mat_16b((p_inp + (itr_1 * input_height * input_width)), input_height, input_width, input_height, input_width, name);
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_bias_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_16b_adv((p_bias + (tmp * 1 * 1)), 1, 1, 1, 1, name);
    }
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_out_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_16b_adv((p_out + (tmp * out_height * out_width)), out_height, out_width, out_height, out_width, name);
    }
  }

}

__attribute__((unused))
static void print_padded_inp_mat_8b
  (pWORD8 p_inp
   ,WORD32 kernel_height
   ,WORD32 kernel_width
   ,WORD32 input_height
   ,WORD32 input_width
   ,WORD32 output_height
   ,WORD32 output_width
   ,WORD32 y_stride
   ,WORD32 x_stride
   ,WORD32 y_padding
   ,WORD32 x_padding
   ,WORD32 input_channels
  )
{
  int i, j;
  int itr_1;
  pWORD8 _pWORD8_inp;
  char name[60] = {0};

  DP_1("==> Imagine how padded input data would look like");

  WORD32 extra_y_padding;
  WORD32 extra_x_padding;

  extra_y_padding =
    y_stride *
      (output_height - RESULT_DIM_OF_CONV(y_padding, input_height, kernel_height, y_stride));

  extra_x_padding =
    x_stride *
      (output_width - RESULT_DIM_OF_CONV(x_padding, input_width, kernel_width, x_stride));

  WORD32 input_height_pad = (y_padding + input_height + extra_y_padding);
  WORD32 input_width_pad  = (x_padding + input_width + extra_x_padding);

  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    _pWORD8_inp = (p_inp + (itr_1 * input_height * input_width));

    sprintf(name, "padded_p_inp_for_inp-ch_%d", itr_1);

    DP_1("\r\n===========[%s][%s][%p]====|%dx%d|=======\r\n", __func__, name, (void *) _pWORD8_inp, input_height_pad, input_width_pad);

    for (i = 0; i < y_padding; i++)
    {
      printf("\r\n|Row = %d|", i);
      for (j = 0; j < input_width_pad; j++)
      {
        DISP_DEC_I8(0);
        printf("|");
      }
      printf("\r\n");
    }
    for (i = 0; i < input_height; i++)
    {
      printf("\r\n|Row = %d|", (i + y_padding));
      for (j = 0; j < x_padding; j++)
      {
        DISP_DEC_I8(0);
        printf("|");
      }
      for (j = 0; j < input_width; j++)
      {
        DISP_DEC_I8(_pWORD8_inp[(i * input_width + j)]);
        printf("|");
      }
      for (j = 0; j < extra_x_padding; j++)
      {
        DISP_DEC_I8(0);
        printf("|");
      }
      printf("\r\n");
    }
    for (i = 0; i < extra_y_padding; i++)
    {
      printf("\r\n|Row = %d|", (i + y_padding + input_height));
      for (j = 0; j < input_width_pad; j++)
      {
        DISP_DEC_I8(0);
        printf("|");
      }
      printf("\r\n");
    }
  }
}

__attribute__((unused))
static void dbg_convds_8x8_8
  (pWORD8 p_out
  ,pWORD8 p_kernel
  ,pWORD8 p_inp
  ,pWORD8 p_bias
  ,WORD32 input_height
  ,WORD32 input_width
  ,WORD32 input_channels
  ,WORD32 kernel_height
  ,WORD32 kernel_width
  ,WORD32 channels_multiplier
  ,WORD32 out_height
  ,WORD32 out_width
  )
{
  int itr_1;
  int itr_2;
  int tmp;
  char name[60] = {0};

  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_kernel_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_8b_adv((p_kernel + (tmp * kernel_height * kernel_width)), kernel_height, kernel_width, kernel_height, kernel_width, name);
    }
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    sprintf(name, "p_inp_for_inp-ch_%d", itr_1);
    print_mat_8b((p_inp + (itr_1 * input_height * input_width)), input_height, input_width, input_height, input_width, name);
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_bias_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_8b_adv((p_bias + (tmp * 1 * 1)), 1, 1, 1, 1, name);
    }
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_out_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_8b_adv((p_out + (tmp * out_height * out_width)), out_height, out_width, out_height, out_width, name);
    }
  }

}

__attribute__((unused))
static void dbg_convds_16x16_16
  (pWORD16 p_out
  ,pWORD16 p_kernel
  ,pWORD16 p_inp
  ,pWORD16 p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  out_height
  ,WORD32  out_width
  )
{
  int itr_1;
  int itr_2;
  int tmp;
  char name[60] = {0};

  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_kernel_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_16b_adv((p_kernel + (tmp * kernel_height * kernel_width)), kernel_height, kernel_width, kernel_height, kernel_width, name);
    }
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    sprintf(name, "p_inp_for_inp-ch_%d", itr_1);
    print_mat_16b((p_inp + (itr_1 * input_height * input_width)), input_height, input_width, input_height, input_width, name);
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_bias_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_16b_adv((p_bias + (tmp * 1 * 1)), 1, 1, 1, 1, name);
    }
  }
  for (itr_1 = 0; itr_1 < input_channels; itr_1++)
  {
    for (itr_2 = 0; itr_2 < channels_multiplier; itr_2++)
    {
      tmp = (itr_1* channels_multiplier + itr_2);
      sprintf(name, "p_out_for_inp-ch_%d_ch-mult_%d_total_%d", itr_1, itr_2, tmp);
      print_mat_16b_adv((p_out + (tmp * out_height * out_width)), out_height, out_width, out_height, out_width, name);
    }
  }

}

#define UNSIGNED_TYPE 0
#define SIGNED_TYPE 1

__attribute__((unused))
  static void print_mat(void *mat, int type, int precision, int rows, int cols, int rows_to_show, int cols_to_show, char *mat_name)
{
  switch (type)
  {
    case SIGNED_TYPE:
      switch (precision)
      {
        case 8:
          {
            DP_1("Not supported, type = %d, precision = %d", type, precision);
            break;
          }
        case 16:
          {
            print_mat_16b_adv((pWORD16) mat, rows, cols, rows_to_show, cols_to_show, mat_name);
            break;
          }
        case 32:
          {
            DP_1("Not supported, type = %d, precision = %d", type, precision);
            break;
          }
        case 64:
          {
            DP_1("Not supported, type = %d, precision = %d", type, precision);
            break;
          }
        case -1:
          {
            /* Use this case for float. Ignore the case number. Its just for
             * distinction from others. */
            DP_1("Not supported, type = %d, precision = %d", type, precision);
            break;
          }
        default:
          {
            printf("Error: Invalid value [%d]\r\n", precision);
            break;
          }
      }
      break;
    case UNSIGNED_TYPE:
      DP_1("Not supported, type = %d, precision = %d", type, precision);
      break;
  }
}


#else /* !API_DEBUG */

#define DISP_HEX_16x4(...)

#define DBG_8x8_8_activation(...)
#define DBG_u8xu8_u8(...)
#define DBG_8x8_8(...)
#define DBG_8x8_16(...)
#define DBG_8x8_32(...)
#define DBG_8x16_16_activation(...)
#define DBG_8x16_16(...)
#define DBG_8x16_32(...)
#define DBG_8x16_64(...)
#define DBG_16x16_16_activation(...)
#define DBG_16x16_16(...)
#define DBG_16x16_32(...)
#define DBG_16x16_64(...)
#define DBG_f32xf32_f32(...)
#define DBG_f32xf32_f32_activation(...)

#define PRINT_HEX_STR(...)

#define DISP_HEX_I8(...)
#define DISP_HEX_UI16(...)
#define DISP_HEX_I16(...)
#define DISP_HEX_I32(...)
#define DISP_HEX_I64(...)

#define PRINT_HEX_UI8(...)
#define PRINT_HEX_I8(...)
#define PRINT_HEX_UI16(...)
#define PRINT_HEX_I16(...)
#define PRINT_HEX_I32(...)
#define PRINT_HEX_I64(...)

#define DISP_DEC_I8(...)
#define DISP_DEC_I16(...)
#define DISP_DEC_I32(...)
#define DISP_DEC_I64(...)

#define PRINT_DEC_UI8(...)
#define PRINT_DEC_I8(...)
#define PRINT_DEC_UI16(...)
#define PRINT_DEC_I16(...)
#define PRINT_DEC_I32(...)
#define PRINT_DEC_I64(...)

/* Don't do anything in release builds */
#define DP_1(...)
#define DP_2(...)
#define DP_3(...)
#define DP_4(...)
#define DP_5(...)

#define DBG_SELF_INFO {}
#define PTR_ALIGNMENT_STATUS(...)

#define RETURN_IF_ANY_NULL_FOUND_2(...)
#define RETURN_IF_ANY_NULL_FOUND_3(...)
#define RETURN_IF_ALL_PAIRS_FOUND_NULL_4(...)

#endif /* API_DEBUG */

#endif /* __XA_NNLIB_DEBUG_H__ */
