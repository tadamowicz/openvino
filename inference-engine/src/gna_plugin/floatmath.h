//
// Copyright 2012-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma  once

#include <stdlib.h>
#include <stdio.h>
#ifndef _NO_MKL_
#include <mkl_dnn.h>
#include <mkl_cblas.h>
#endif
// #include "types.h"
#include "dnn.h"

#ifndef CBLAS_LAYOUT
#define CBLAS_LAYOUT CBLAS_ORDER
#endif

#define CNN_MAX_POOL_SIZE 6

void CNNFilter32(intel_dnn_component_t *component);
void CNNMaxPool(intel_dnn_component_t *component, intel_dnn_number_type_t number_type);

#ifdef _NO_MKL_
#ifndef _MKL_H_
#define _MKL_H_
typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;
typedef CBLAS_LAYOUT CBLAS_ORDER; /* this for backward compatibility with CBLAS_ORDER */
#define MKL_INT int
#endif  // #ifndef _MKL_H_
#endif  // #ifdef _NO_MKL_

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif

#ifdef _NO_MKL_
void cblas_sgemm1(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                  const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                  const MKL_INT K, const float alpha, const float *A,
                  const MKL_INT lda, const float *B, const MKL_INT ldb,
                  const float beta, float *C, const MKL_INT ldc);
void cblas_ssbmv1(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const MKL_INT N, const MKL_INT K, const float alpha, const float *A,
                  const MKL_INT lda, const float *X, const MKL_INT incX,
                  const float beta, float *Y, const MKL_INT incY);
#endif  // #ifdef _NO_MKL_
void cblas_sgemm_subset(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                        const MKL_INT K, const float alpha, const float *A,
                        const MKL_INT lda, const float *B, const MKL_INT ldb,
                        const float beta, float *C, const MKL_INT ldc,
                        const uint32_t *OutputList, const MKL_INT L);
void sgemv_split(const uint32_t N,
                 const uint32_t K1,
                 const uint32_t K2,
                 const float *A1,
                 const float *A2,
                 const float *X,
                 const float *B,
                 float *C);

#ifdef __cplusplus
}
#endif

