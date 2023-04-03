/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
# if defined(__APPLE__) && defined(__arm64__)
#include <pthread.h>
# endif

#if 0
#define UPFRONT_SPLIT
#endif
#if 0
#define INLINE_SPLIT_BREAK_DEP
#endif

typedef struct gemm_def {
  libxsmm_datatype in_type;
  libxsmm_datatype out_type;
  libxsmm_datatype comp_type;
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  int vnni_a;
  int order;
} gemm_def;

LIBXSMM_INLINE
double get_random_posneg_p5_num(void) {
  double tmp = libxsmm_rng_f64()-0.5;

  if ( tmp < -0.4 ) {
    tmp = -0.4;
  } else if ( tmp < -0.3 ) {
    tmp = -0.3;
  } else if ( tmp < -0.2 ) {
    tmp = -0.2;
  } else if ( tmp < -0.1 ) {
    tmp = -0.1;
  } else if ( tmp < 0 ) {
    tmp = 0;
  } else if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else {
    tmp = 0.5;
  }

  return tmp;
}

LIBXSMM_INLINE
double get_random_pos_p5_num(void) {
  double tmp = libxsmm_rng_f64();

  if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else if ( tmp < 0.6 ) {
    tmp = 0.6;
  } else if ( tmp < 0.7 ) {
    tmp = 0.7;
  } else if ( tmp < 0.8 ) {
    tmp = 0.8;
  } else if ( tmp < 0.9 ) {
    tmp = 0.9;
  } else if ( tmp < 1.0 ) {
    tmp = 1.0;
  } else {
    tmp = 1.0;
  }

  return tmp;
}

LIBXSMM_INLINE
void init_random_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n, const libxsmm_blasint pos_val_only ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf16_data = (libxsmm_bfloat16*) data;
  libxsmm_bfloat8* bf8_data = (libxsmm_bfloat8*) data;
  libxsmm_hfloat8* hf8_data = (libxsmm_hfloat8*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* sc_data = (char*) data;
  unsigned char* uc_data = (unsigned char*) data;
  libxsmm_blasint l_r, l_i, l_j;

  for (l_r = 0; l_r < br; l_r++) {
    for (l_i = 0; l_i < ld; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        if ( dtype == LIBXSMM_DATATYPE_F64 ) {
          d_data[(l_r * ld * n) + (l_j * ld) + l_i] = (pos_val_only > 0 ) ? get_random_pos_p5_num() :  get_random_posneg_p5_num();
        } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
          f_data[(l_r * ld * n) + (l_j * ld) + l_i] = (pos_val_only > 0 ) ? (float)get_random_pos_p5_num() : (float)get_random_posneg_p5_num();
        } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
          libxsmm_bfloat16_f32 tmp /*= { 0 }*/;
          tmp.f = (pos_val_only > 0 ) ? (float)get_random_pos_p5_num() : (float)get_random_posneg_p5_num();
          bf16_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
          union libxsmm_bfloat8_f16 tmp/* = { 0 }*/;
          tmp.hf = libxsmm_convert_f32_to_f16( (float)get_random_posneg_p5_num() );
          bf8_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_HF8 ) {
          float tmp_rnd = (float)get_random_posneg_p5_num();
          libxsmm_rne_convert_fp32_hf8( &tmp_rnd, &hf8_data[(l_r * ld * n) + (l_j * ld) + l_i], 1 );
        } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
          i_data[(l_r * ld * n) + (l_j * ld) + l_i] = (int)  (get_random_posneg_p5_num() * 40.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I16 ) {
          s_data[(l_r * ld * n) + (l_j * ld) + l_i] = (short)(get_random_posneg_p5_num() * 40.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
          if ( pos_val_only != 0 ) {
            uc_data[(l_r * ld * n) + (l_j * ld) + l_i] = (unsigned char) (get_random_pos_p5_num() * 20.0);
          } else {
            sc_data[(l_r * ld * n) + (l_j * ld) + l_i] = (char) (get_random_posneg_p5_num() * 40.0);
          }
        } else {
        }
      }
    }
  }
}

LIBXSMM_INLINE
void init_zero_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0x0, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

LIBXSMM_INLINE
void init_garbage_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0xdeadbeef, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

LIBXSMM_INLINE
double get_flops( const libxsmm_datatype dtype, const int i_order ) {
  double result = 0.0;

  if ( i_order == 5 ) {
    if ( dtype == LIBXSMM_DATATYPE_F32 ) {
      result = 35.0 * 9.0 * 20.0 * 3.0 * 2.0;
    } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
      result = 128.0 * 9.0 * 64.0 * 2.0;
    } else {
      /* shouldn't happen */
    }
  } else {
    /* shouldn't happen */
  }

  return result;
}

LIBXSMM_INLINE
double jit_matmul_f32( const gemm_def*    i_gemm_def,
                       const int          i_reps,
                       const int          i_elems ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
#if 0
  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };
#endif
  libxsmm_timer_tickint l_start;
  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_gemm_param gemm_param;

  double l_runtime;
  size_t l_e, l_r;
#if 0
  int l_cfg_flags = 0;
  int l_rls_flags = 0;
#endif
  char *l_a, *l_b, *l_c;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_a = NULL;
  l_b = NULL;
  l_c = NULL;
  l_a = (char*)libxsmm_aligned_malloc((size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * 3 * LIBXSMM_TYPESIZE(i_gemm_def->in_type), 64);
  l_b = (char*)libxsmm_aligned_malloc((size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * (size_t)i_elems * LIBXSMM_TYPESIZE(i_gemm_def->in_type), 64);
  l_c = (char*)libxsmm_aligned_malloc((size_t)i_gemm_def->ldc * (size_t)i_gemm_def->n * (size_t)i_elems * LIBXSMM_TYPESIZE(i_gemm_def->out_type), 64);

  init_random_matrix( i_gemm_def->in_type,  l_a, 3,       i_gemm_def->lda, i_gemm_def->k, 0 );
  init_random_matrix( i_gemm_def->in_type,  l_b, i_elems, i_gemm_def->ldb, i_gemm_def->n, 0 );
  init_zero_matrix(   i_gemm_def->out_type, l_c, i_elems, i_gemm_def->ldc, i_gemm_def->n );

  /* set up the flags */
  if ( i_gemm_def->vnni_a != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }

  /* setting update GEMM struct */
  l_shape = libxsmm_create_gemm_shape( i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
      i_gemm_def->lda, i_gemm_def->ldb, i_gemm_def->ldc,
      i_gemm_def->in_type, i_gemm_def->in_type, i_gemm_def->out_type, i_gemm_def->comp_type );

#if 0
  if (i_gemm_def->tc_config) {
    l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
    l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
    l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
    cfg_tr.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_cfg_flags, l_prefetch_flags, l_brconfig );
    rls_tr.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_rls_flags, l_prefetch_flags, l_brconfig );
  }
#endif
  l_test_jit.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
  if (l_test_jit.xmm == NULL) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  /* run external tileconfig */
#if 0
  if (i_gemm_def->tc_config) {
    cfg_tr.gemm( NULL );
  }
#endif

  /* reset GEMM parameter */
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

  /* run performance */
  l_start = libxsmm_timer_tick();
  for (l_r = 0; l_r < (size_t)i_reps; l_r++) {
    for ( l_e = 0; l_e < (size_t)i_elems; l_e++) {
      gemm_param.a.primary = (void*)l_a;
      gemm_param.b.primary = (void*)(l_b + ((size_t)l_e * i_gemm_def->ldb * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type)));
      gemm_param.c.primary = (void*)(l_c + ((size_t)l_e * i_gemm_def->ldc * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->out_type)));
      l_test_jit.gemm( &gemm_param );
      gemm_param.a.primary = (void*)(l_a + ((size_t)i_gemm_def->lda * i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type)));
      l_test_jit.gemm( &gemm_param );
      gemm_param.a.primary = (void*)(l_a + ((size_t)i_gemm_def->lda * i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type) * 2));
      l_test_jit.gemm( &gemm_param );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  /* run external tilerelease */
#if 0
  if (i_gemm_def->tc_config) {
    rls_tr.gemm( NULL );
  }
#endif

  libxsmm_free( (void*)l_a );
  libxsmm_free( (void*)l_b );
  libxsmm_free( (void*)l_c );

  return l_runtime;
}

LIBXSMM_INLINE
double jit_matmul_bf16( const gemm_def*    i_gemm_def,
                        const int          i_reps,
                        const int          i_elems ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_gemm_shape l_shape;
#if 1
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
#else
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
#endif
  libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_gemm_param gemm_param;

  libxsmm_meltw_unary_shape split_shape = libxsmm_create_meltw_unary_shape( i_gemm_def->k, i_gemm_def->n, i_gemm_def->ldb, i_gemm_def->ldb, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltw_unary_flags split_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type split_type = LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3;
  libxsmm_meltwfunction_unary split_kernel;
  libxsmm_meltw_unary_param split_param /*= { 0 }*/;

  libxsmm_meltw_unary_shape copy_shape = libxsmm_create_meltw_unary_shape( i_gemm_def->m, i_gemm_def->n, i_gemm_def->ldc, i_gemm_def->ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  libxsmm_meltw_unary_flags copy_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type copy_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  libxsmm_meltwfunction_unary copy_kernel;
  libxsmm_meltw_unary_param copy_param /*= { 0 }*/;

  double l_runtime;
  size_t l_e, l_r;
  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  libxsmm_blasint l_lda, l_ldb, l_ldc, l_m, l_n, l_k;
  char *l_a, *l_b, *l_c, *l_c_bf16;
#ifdef UPFRONT_SPLIT
  char *l_b_bf16_elems;
#else
  char *l_b_bf16;
#ifdef INLINE_SPLIT_BREAK_DEP
  char *l_b_bf16_two;
#endif
#endif
  unsigned long long strides[2];

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  if ( i_gemm_def->order == 5 ) {
    l_lda = 128;
    l_ldb = 64;
    l_ldc = 128;
    l_m = 128;
    l_n = 9;
    l_k = 64;
  }

  l_a = NULL;
  l_b = NULL;
  l_c = NULL;

  l_a = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * LIBXSMM_TYPESIZE(i_gemm_def->in_type), 64);
  l_b = (char*)libxsmm_aligned_malloc((size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * (size_t)i_elems * LIBXSMM_TYPESIZE(i_gemm_def->out_type), 64);
  l_c = (char*)libxsmm_aligned_malloc((size_t)i_gemm_def->ldc * (size_t)i_gemm_def->n * (size_t)i_elems * LIBXSMM_TYPESIZE(i_gemm_def->out_type), 64);
#ifdef UPFRONT_SPLIT
  l_b_bf16_elems = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)i_gemm_def->n * (size_t)i_elems * LIBXSMM_TYPESIZE(i_gemm_def->in_type), 64);
#else
  l_b_bf16 = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type), 64);
#ifdef INLINE_SPLIT_BREAK_DEP
  l_b_bf16_two = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type), 64);
#endif
#endif
  l_c_bf16 = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->out_type), 64);

  init_random_matrix( i_gemm_def->in_type,  l_a, 1,       l_lda, l_k, 0 );
  init_random_matrix( i_gemm_def->out_type, l_b, i_elems, i_gemm_def->ldb, i_gemm_def->n, 0 );
  init_zero_matrix(   i_gemm_def->out_type, l_c, i_elems, i_gemm_def->ldc, i_gemm_def->n );
#ifdef UPFRONT_SPLIT
  init_zero_matrix(   i_gemm_def->in_type,  l_b_bf16_elems, i_elems, l_ldb, i_gemm_def->n );
#else
  init_zero_matrix(   i_gemm_def->in_type,  l_b_bf16, 1, l_ldb, i_gemm_def->n );
#ifdef INLINE_SPLIT_BREAK_DEP
  init_zero_matrix(   i_gemm_def->in_type,  l_b_bf16_two, 1, l_ldb, i_gemm_def->n );
#endif
#endif
  init_zero_matrix(   i_gemm_def->in_type,  l_c_bf16, 1, l_ldc, i_gemm_def->n );

  /* set up the flags */
  if ( i_gemm_def->vnni_a != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }

  /* setting update GEMM struct */
  l_shape = libxsmm_create_gemm_shape( l_m, l_n, l_k,
      l_lda, l_ldb, l_ldc,
      i_gemm_def->in_type, i_gemm_def->in_type, i_gemm_def->out_type, i_gemm_def->comp_type );

  l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
  l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  cfg_tr.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_cfg_flags, l_prefetch_flags );
  rls_tr.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_rls_flags, l_prefetch_flags );

  l_test_jit.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
  if (l_test_jit.xmm == NULL) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  split_kernel = libxsmm_dispatch_meltw_unary_v2( split_type, split_shape, split_flags );
  if (split_kernel  == NULL) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }
  copy_kernel  = libxsmm_dispatch_meltw_unary_v2( copy_type, copy_shape, copy_flags );
  if (copy_kernel  == NULL) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  /* run external tileconfig */
  cfg_tr.gemm( NULL );

  /* reset GEMM parameter */
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
  memset( &split_param, 0, sizeof(libxsmm_meltw_unary_param) );
  memset( &copy_param, 0, sizeof(libxsmm_meltw_unary_param) );
  strides[0] = (unsigned long long)(LIBXSMM_TYPESIZE(i_gemm_def->in_type)*i_gemm_def->ldb*i_gemm_def->n);
  strides[1] = (unsigned long long)(LIBXSMM_TYPESIZE(i_gemm_def->in_type)*i_gemm_def->ldb*i_gemm_def->n*2);

#ifdef UPFRONT_SPLIT
  for ( l_e = 0; l_e < (size_t)i_elems; l_e++) {
    split_param.in.primary = (void*)(l_b + ((size_t)l_e * i_gemm_def->ldb * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->out_type)));
    split_param.out.primary = (void*)(l_b_bf16_elems + ((size_t)l_e * l_ldb * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type)));
    split_param.out.secondary = (void*)(&strides[0]);
    split_kernel( &split_param );
  }
#else
#ifdef INLINE_SPLIT_BREAK_DEP
  /* we need some resonable split (bit toggling) */
  split_param.in.primary = (void*)l_b;
  split_param.out.primary = (void*)l_b_bf16;
  split_param.out.secondary = (void*)(&strides[0]);
  split_kernel( &split_param );
#endif
#endif

    /* run performance */
  l_start = libxsmm_timer_tick();
  for (l_r = 0; l_r < (size_t)i_reps; l_r++) {
   for ( l_e = 0; l_e < (size_t)i_elems; l_e++) {
#ifdef UPFRONT_SPLIT
      gemm_param.b.primary = (void*)(l_b_bf16_elems + ((size_t)l_e* l_ldb * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type)));
#else
      split_param.in.primary = (void*)(l_b + ((size_t)l_e * i_gemm_def->ldb * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->out_type)));
#ifdef INLINE_SPLIT_BREAK_DEP
      split_param.out.primary = (void*)l_b_bf16_two;
#else
      split_param.out.primary = (void*)l_b_bf16;
#endif
      split_param.out.secondary = (void*)(&strides[0]);
      split_kernel( &split_param );
      gemm_param.b.primary = (void*)l_b_bf16;
#endif
      gemm_param.a.primary = (void*)l_a;
      gemm_param.c.primary = (void*)l_c_bf16;
      l_test_jit.gemm( &gemm_param );

      copy_param.in.primary = (void*)l_c_bf16;
      copy_param.out.primary = (void*)(l_c + ((size_t)l_e * i_gemm_def->ldc * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->out_type)));
      copy_kernel( &copy_param );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  /* run external tilerelease */
  rls_tr.gemm( NULL );

  libxsmm_free( (void*)l_a );
  libxsmm_free( (void*)l_b );
  libxsmm_free( (void*)l_c );
#ifdef UPFRONT_SPLIT
  libxsmm_free( (void*)l_b_bf16_elems );
#else
  libxsmm_free( (void*)l_b_bf16 );
#ifdef INLINE_SPLIT_BREAK_DEP
  libxsmm_free( (void*)l_b_bf16_two );
#endif
#endif
  libxsmm_free( (void*)l_c_bf16 );

  return l_runtime;
}

LIBXSMM_INLINE
void print_help(void) {
  printf("\n\n");
  printf("1. Usage:\n");
  printf("    Order\n");
  printf("    #elems\n");
  printf("    PRECISION: F32, BF16F32\n");
  printf("\n\n");
}

int main(int argc, char* argv []) {
  char* l_precision = NULL;
  double l_runtime_libxsmm = 0;
  int l_reps;
  int l_order;
  int l_elems;
  gemm_def l_gemm_def;
  int l_n_threads = 1;

  /* check argument count for a valid range */
  if ( argc == 5 ) {
    /* xgemm sizes */
    l_order = atoi(argv[1]);
    l_elems = atoi(argv[2]);
    l_precision = argv[3];
    l_reps = atoi(argv[4]);
  } else {
    print_help();
    return EXIT_FAILURE;
  }

  /* setting static GEMM parameters */
  l_gemm_def.order = l_order;
  if ( l_order == 5 ) {
    l_gemm_def.m = 35;
    l_gemm_def.n = 9;
    l_gemm_def.k = 20;
    l_gemm_def.lda = 35;
    l_gemm_def.ldb = 20;
    l_gemm_def.ldc = 35;
  } else {
    fprintf(stderr, "Unsupported order %i!\n", l_order);
    exit(EXIT_FAILURE);
  }

  /* setting precision in GEMM struct */
  if ( (strcmp(l_precision, "F32") == 0) ) {
    l_gemm_def.vnni_a = 0;
    l_gemm_def.in_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
  } else if (strcmp(l_precision, "BF16F32") == 0) {
    l_gemm_def.vnni_a = 1;
    l_gemm_def.in_type = LIBXSMM_DATATYPE_BF16;
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
    l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;
  } else {
    fprintf(stderr, "Unsupported precision %s!\n", l_precision);
    exit(EXIT_FAILURE);
  }

  /* read the number of threads */
#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
# pragma omp parallel
  {
#   pragma omp master
    {
      l_n_threads = omp_get_num_threads();
    }
  }
#else
  l_n_threads = 1;
#endif
  /* set rng seed */
  libxsmm_rng_set_seed( 555 );

#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
# pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
  {
    /* run LIBXSMM solution */
    if ( (strcmp(l_precision, "F32") == 0) ) {
      l_runtime_libxsmm = jit_matmul_f32( &l_gemm_def, l_reps, l_elems );
    } else {
      l_runtime_libxsmm = jit_matmul_bf16( &l_gemm_def, l_reps, l_elems );
    }
  } /* close parallel region */

  printf("%i number of threads\n", l_n_threads);
  printf("%fs for libxsmm\n", l_runtime_libxsmm/(double)l_n_threads);
  if ( (strcmp(l_precision, "F32") == 0) ) {
    printf("%f GFLOPS for libxsmm F32\n", ((double)((double)l_reps * get_flops( l_gemm_def.in_type, l_order ) * (double)l_elems * (double)l_n_threads) / (l_runtime_libxsmm * 1.0e9)));
  } else {
    printf("%f GFLOPS for libxsmm\n", ((double)((double)l_reps * get_flops( l_gemm_def.in_type, l_order ) * (double)l_elems * (double)l_n_threads) / (l_runtime_libxsmm * 1.0e9)));
  }
}
