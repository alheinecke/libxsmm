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
  double alpha;
  double beta;
  int vnni_a;
  int br_type;
  libxsmm_blasint br_count;
  int br_unroll;
} gemm_def;

LIBXSMM_INLINE
double get_random_posneg_p5_num(void) {
  double tmp = libxsmm_rng_f64()-0.5;

#if 0
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
#endif

  return tmp;
}

LIBXSMM_INLINE
double get_random_pos_p5_num(void) {
  double tmp = libxsmm_rng_f64();

#if 0
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
#endif

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
void convert_a_to_vnni2( gemm_def* i_gemm_def, void* l_a_flat, void* l_a_vnni ) {
  libxsmm_blasint l_i, l_j, l_i2;
  libxsmm_blasint lda = i_gemm_def->lda;
  libxsmm_blasint m = i_gemm_def->m;
  libxsmm_blasint n = i_gemm_def->k;

  if (i_gemm_def->in_type == LIBXSMM_DATATYPE_BF16) {
    libxsmm_bfloat16* i_a = (libxsmm_bfloat16*)l_a_flat;
    libxsmm_bfloat16* o_a = (libxsmm_bfloat16*)l_a_vnni;
    /* convert to vnni */
    for (l_i = 0; l_i < n/2; l_i++) {
      for (l_j = 0; l_j < m; l_j++) {
        for (l_i2 = 0; l_i2 < 2; l_i2++) {
          o_a[(l_i*lda*2)+(l_j*2)+l_i2] = i_a[(((l_i*2)+l_i2)*lda)+l_j];
        }
      }
    }
  } else {
    /* Should not happen */
  }
}

LIBXSMM_INLINE
double check_matrix( const libxsmm_datatype dtype, const void* data_gold, const void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  libxsmm_matdiff_info l_diff;
  double error = 0.0;

  libxsmm_matdiff_clear(&l_diff);
  if ( dtype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
#if 0
    float* data_gold_f = (float*)data_gold;
    float* data_f      = (float*)data;
    libxsmm_blasint l_i, l_j;

    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        printf("gold: %10.10f, computed: %10.10f, diff: %10.10f\n", data_gold_f[(l_j * ld) + l_i], data_f[(l_j * ld) + l_i], data_gold_f[(l_j * ld) + l_i]-data_f[(l_j * ld) + l_i] );
      }
    }
#endif
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
    float* data_gold_f = (float*)malloc( sizeof(float) * ld * n );
    float* data_f      = (float*)malloc( sizeof(float) * ld * n );

    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);

    free( data_f );
    free( data_gold_f ) ;
  } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
    float* data_gold_f = malloc( ld * n * sizeof(float) );
    float* data_f      = malloc( ld * n * sizeof(float) );

    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = l_diff.normf_rel;

    free( data_f );
    free( data_gold_f ) ;
  } else if ( dtype == LIBXSMM_DATATYPE_HF8 ) {
    float* data_gold_f = malloc( ld * n * sizeof(float) );
    float* data_f      = malloc( ld * n * sizeof(float) );

    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = l_diff.normf_rel;

#if 0
    libxsmm_blasint l_i, l_j;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        printf("gold: %f, computed: %f\n", data_gold_f[(l_j * ld) + l_i], data_f[(l_j * ld) + l_i] );
      }
    }
#endif

    free( data_f );
    free( data_gold_f ) ;
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_I32, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_I8, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else {
    error = 100.0;
  }

  printf("\nPrinting Norms:\n");
  printf("L1 reference  : %.25g\n", l_diff.l1_ref);
  printf("L1 test       : %.25g\n", l_diff.l1_tst);
  printf("L2 abs.error  : %.24f\n", l_diff.l2_abs);
  printf("L2 rel.error  : %.24f\n", l_diff.l2_rel);
  printf("Linf abs.error: %.24f\n", l_diff.linf_abs);
  printf("Linf rel.error: %.24f\n", l_diff.linf_rel);
  printf("Check-norm    : %.24f\n", error);
  printf("\n");

  return error;
}

LIBXSMM_INLINE
double jit_matmul( const gemm_def*    i_gemm_def,
                   const void*        i_a,
                   const void*        i_b,
                   void*              o_c,
                   void*              o_c_perf,
                   const int          i_reps ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_gemm_param gemm_param;
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  char** l_a_addr = (char**)malloc(sizeof(char*)*i_gemm_def->br_count);
  char** l_b_addr = (char**)malloc(sizeof(char*)*i_gemm_def->br_count);
  long long* l_a_offs = (long long*)malloc(sizeof(long long)*i_gemm_def->br_count);
  long long* l_b_offs = (long long*)malloc(sizeof(long long)*i_gemm_def->br_count);
  double l_beta = i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < (size_t)i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (long long)i_gemm_def->lda * i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
      l_b_offs[l_r] = l_r * (long long)i_gemm_def->ldb * i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->vnni_a != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
  }

  l_flags |= ( l_beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;

  /* setting update GEMM struct */
  l_shape = libxsmm_create_gemm_shape( i_gemm_def->m,  i_gemm_def->n, i_gemm_def->k,
      i_gemm_def->lda, i_gemm_def->ldb, i_gemm_def->ldc,
      i_gemm_def->in_type, i_gemm_def->in_type, i_gemm_def->out_type, i_gemm_def->comp_type );

  /* setting BRGEMM config struct */
  if (i_gemm_def->br_type == 1) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = (unsigned char)(( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count);
  } else if (i_gemm_def->br_type == 2) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = (unsigned char)(( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count);
  } else if (i_gemm_def->br_type == 3) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = i_gemm_def->lda*i_gemm_def->k*LIBXSMM_TYPESIZE(i_gemm_def->in_type);
    l_brconfig.br_stride_b_hint = i_gemm_def->ldb*i_gemm_def->n*LIBXSMM_TYPESIZE(i_gemm_def->in_type);
    l_brconfig.br_unroll_hint = (unsigned char)(( i_gemm_def->br_unroll == 0 ) ? 0 : i_gemm_def->br_count);
  } else {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = 0;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit.gemm = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (l_test_jit.xmm == NULL) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  /* reset GEMM parameter */
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

  gemm_param.op.tertiary = &l_br;
  gemm_param.c.primary = (void*)o_c;
  gemm_param.c.tertiary = NULL;
  /* run correctness */
  if (i_gemm_def->br_type == 0) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    if ( l_info.prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      gemm_param.a.quaternary = (void*)i_a;
      gemm_param.b.quaternary = (void*)i_b;
      gemm_param.c.quaternary = (void*)o_c;
    }
    l_test_jit.gemm( &gemm_param );
  } else if (i_gemm_def->br_type == 1) {
    gemm_param.a.primary = l_a_addr;
    gemm_param.b.primary = l_b_addr;
    for ( l_r = 0 ; l_r < (size_t)i_gemm_def->br_count; l_r++ ) {
      l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
      l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
    }
    l_test_jit.gemm( &gemm_param );
  } else if (i_gemm_def->br_type == 2) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.a.secondary = l_a_offs;
    gemm_param.b.primary = (void*)i_b;
    gemm_param.b.secondary = l_b_offs;
    l_test_jit.gemm( &gemm_param );
  } else if (i_gemm_def->br_type == 3) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    l_test_jit.gemm( &gemm_param );
  }

  /* run performance */
  gemm_param.c.primary = (void*)o_c_perf;
  l_start = libxsmm_timer_tick();
  if (i_gemm_def->br_type == 0) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    if ( l_info.prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      gemm_param.a.quaternary = (void*)i_a;
      gemm_param.b.quaternary = (void*)i_b;
      gemm_param.c.quaternary = (void*)o_c_perf;
    }
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
      l_test_jit.gemm( &gemm_param );
    }
  } else if (i_gemm_def->br_type == 1) {
    gemm_param.a.primary = l_a_addr;
    gemm_param.b.primary = l_b_addr;
    assert(NULL != l_a_addr && NULL != l_b_addr);
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
      for ( l_r = 0 ; l_r < (size_t)i_gemm_def->br_count; l_r++ ) {
        l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
        l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * LIBXSMM_TYPESIZE(i_gemm_def->in_type));
      }
      l_test_jit.gemm( &gemm_param );
    }
  } else if (i_gemm_def->br_type == 2) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.a.secondary = l_a_offs;
    gemm_param.b.primary = (void*)i_b;
    gemm_param.b.secondary = l_b_offs;
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
      l_test_jit.gemm( &gemm_param );
    }
  } else if (i_gemm_def->br_type == 3) {
    gemm_param.a.primary = (void*)i_a;
    gemm_param.b.primary = (void*)i_b;
    for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
      l_test_jit.gemm( &gemm_param );
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
  printf("%fs for creating jit\n", l_jittime);

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}

LIBXSMM_INLINE
void print_help(void) {
  printf("\n\n");
  printf("1. Usage (dense*dense=dense, correctness and performance):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    output high prec: 0 or 1\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("\n\n");
}

int main(int argc, char* argv []) {
  libxsmm_blasint l_lda = 0, l_ldb = 0, l_ldc = 0;
  libxsmm_blasint l_m = 0, l_n = 0, l_k = 0;
  double l_alpha = 0;
  double l_beta = 0;
  int l_br = 1;
  int l_br_type = 0;
  int l_br_unroll = 0;
  double l_runtime_libxsmm_flat = 0;
  double l_runtime_libxsmm = 0;
  double error = 0.0;
  double l_total_max_error = 0.0;
  int l_reps;
  gemm_def l_gemm_def;
  int l_n_threads = 1;
  unsigned int l_i = 0;
  int l_out_high_prec = 0;

# if defined(__APPLE__) && defined(__arm64__)
#  if 1
  pthread_set_qos_class_self_np( QOS_CLASS_USER_INTERACTIVE, 0 );
#  else
  pthread_set_qos_class_self_np( QOS_CLASS_BACKGROUND, 0 );
#  endif
# endif

  /* check argument count for a valid range */
  if ( argc == 14 ) {
    /* xgemm sizes */
    l_m = atoi(argv[1]);
    l_n = atoi(argv[2]);
    l_k = atoi(argv[3]);
    l_lda = atoi(argv[4]);
    l_ldb = atoi(argv[5]);
    l_ldc = atoi(argv[6]);

    /* some sugar */
    l_alpha = atof(argv[7]);
    l_beta = atof(argv[8]);

    /* using f32 as output */
    l_out_high_prec = atoi(argv[9]);

    /* brgemm */
    l_br = atoi(argv[11]);
    l_br_unroll = atoi(argv[12]);
    l_reps = atoi(argv[13]);

    if (strcmp("nobr", argv[10]) == 0) {
      l_br_type = 0;
    }
    else if (strcmp("addrbr", argv[10]) == 0) {
      l_br_type = 1;
    }
    else if (strcmp("offsbr", argv[10]) == 0) {
      l_br_type = 2;
    }
    else if (strcmp("strdbr", argv[10]) == 0) {
      l_br_type = 3;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

  } else {
    print_help();
    return EXIT_FAILURE;
  }

  l_out_high_prec = (l_out_high_prec != 0 ) ? 1 : 0;
  l_br = (l_br < 1) ? 1 : l_br;
  l_br = (l_br_type == 0) ? 1 : l_br;
  l_br_unroll = (l_br_type == 0) ? 0 : l_br_unroll;

  /* check alpha */
  if ( LIBXSMM_NEQ(l_alpha, 1.0) ) {
    fprintf(stderr, "JIT: alpha needs to be 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* check beta */
  if ( LIBXSMM_NEQ(l_beta, 0.0) && LIBXSMM_NEQ(l_beta, 1.0) ) {
    fprintf(stderr, "JIT: beta needs to be 0.0 or 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* setting static GEMM parameters */
  l_gemm_def.alpha = l_alpha;
  l_gemm_def.beta = l_beta;
  l_gemm_def.vnni_a = 0;
  l_gemm_def.br_type = l_br_type;
  l_gemm_def.br_count = l_br;
  l_gemm_def.br_unroll = l_br_unroll;

  /* setting precision in GEMM struct */
  l_gemm_def.in_type = LIBXSMM_DATATYPE_BF16;
  if ( l_out_high_prec != 0 ) {
    l_gemm_def.out_type = LIBXSMM_DATATYPE_F32;
  } else {
    l_gemm_def.out_type = LIBXSMM_DATATYPE_BF16;
  }
  l_gemm_def.comp_type = LIBXSMM_DATATYPE_F32;

  printf("------------------------------------------------\n");
  if ( l_out_high_prec != 0 ) {
    printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i), BF16F32, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_br);
  } else {
    printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i), BF16, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_br);
  }
  printf("------------------------------------------------\n");

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

  l_gemm_def.m = l_m;
  l_gemm_def.n = l_n;
  l_gemm_def.k = l_k;
  l_gemm_def.lda = l_lda;
  l_gemm_def.ldb = l_ldb;
  l_gemm_def.ldc = l_ldc;

  /* set rng seed */
  libxsmm_rng_set_seed( 555 );

#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
# pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
  {
    char *l_a, *l_a_vnni, *l_b, *l_c, *l_c_gold, *l_c_perf;

    l_a_vnni = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
    l_a      = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
    l_b      = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * LIBXSMM_TYPESIZE(l_gemm_def.in_type), 64);
    l_c      = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
    l_c_gold = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);
    l_c_perf = (char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * LIBXSMM_TYPESIZE(l_gemm_def.out_type), 64);

    init_random_matrix( l_gemm_def.in_type, l_a, l_br, l_lda, l_k, 0 );
    for ( l_i = 0; l_i < l_br; ++l_i ) {
      convert_a_to_vnni2(&l_gemm_def, l_a+((size_t)l_i * (size_t)l_lda * (size_t)l_k * LIBXSMM_TYPESIZE(l_gemm_def.in_type)), l_a_vnni+((size_t)l_i * (size_t)l_lda * (size_t)l_k * LIBXSMM_TYPESIZE(l_gemm_def.in_type)));
    }
    init_random_matrix( l_gemm_def.in_type, l_b, l_br, l_ldb, l_n, 0 );

    if ( l_beta == 0 ) {
      init_garbage_matrix( l_gemm_def.out_type, l_c,      1, l_ldc, l_n );
      init_garbage_matrix( l_gemm_def.out_type, l_c_perf, 1, l_ldc, l_n );
      init_garbage_matrix( l_gemm_def.out_type, l_c_gold, 1, l_ldc, l_n );
    } else {
      init_zero_matrix( l_gemm_def.out_type, l_c,      1, l_ldc, l_n );
      init_zero_matrix( l_gemm_def.out_type, l_c_perf, 1, l_ldc, l_n );
      init_zero_matrix( l_gemm_def.out_type, l_c_gold, 1, l_ldc, l_n );
    }

    /* run LIBXSMM solution */
    l_runtime_libxsmm_flat = jit_matmul( &l_gemm_def, l_a,      l_b, l_c_gold, l_c_perf, l_reps );
    l_gemm_def.vnni_a = 1;
    l_runtime_libxsmm      = jit_matmul( &l_gemm_def, l_a_vnni, l_b, l_c,      l_c_perf, l_reps );

    /* run compare */
#if defined(_OPENMP) && defined(LIBXSMM_PARALLEL_KERNEL_TEST)
#   pragma omp master
#endif
    {
      error = check_matrix( l_gemm_def.out_type, l_c_gold, l_c, l_ldc, l_m, l_n );
    }

    libxsmm_free(l_a);
    libxsmm_free(l_a_vnni);
    libxsmm_free(l_b);
    libxsmm_free(l_c);
    libxsmm_free(l_c_perf);
    libxsmm_free(l_c_gold);
  } /* close parallel region */


  printf("%fs for libxsmm\n", l_runtime_libxsmm);
  printf("%f GFLOPS for libxsmm\n", ((double)((double)l_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * (double)l_n_threads * 2.0) / (l_runtime_libxsmm * 1.0e9));
  printf("%fs for libxsmm flat\n", l_runtime_libxsmm_flat);
  printf("%f GFLOPS for libxsmm flat\n", ((double)((double)l_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * (double)l_n_threads * 2.0) / (l_runtime_libxsmm_flat * 1.0e9));
   printf("max. error: %f\n", error);

  if ( l_total_max_error < error ) {
    l_total_max_error = error;
  }

  printf("------------------------------------------------\n");

  /* Print total max error */
  printf("\n\n Total Max Error %f\n\n", l_total_max_error );

  if ( l_total_max_error >= 0.005 ) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}
