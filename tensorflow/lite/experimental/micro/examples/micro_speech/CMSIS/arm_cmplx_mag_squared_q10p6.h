/* This file is a modification of the ARM CMSIS library file arm_math.h
 * We have retained the original copyright and header information, in
 * accordance with the Apache 2.0 license terms.
 */

/******************************************************************************
 * @file     arm_math.h
 * @brief    Public header file for CMSIS DSP LibraryU
 * @version  V1.5.3
 * @date     10. January 2018
 ******************************************************************************/
/*
 * Copyright (C) 2010-2017 ARM Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

void arm_cmplx_mag_squared_q10p6(
  q15_t * pSrc,
  q15_t * pDst,
  uint32_t numSamples);
