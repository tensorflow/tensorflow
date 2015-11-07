// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARALLELIZER_H
#define EIGEN_PARALLELIZER_H

namespace Eigen { 

namespace internal {

/** \internal */
inline void manage_multi_threading(Action action, int* v)
{
  static EIGEN_UNUSED int m_maxThreads = -1;

  if(action==SetAction)
  {
    eigen_internal_assert(v!=0);
    m_maxThreads = *v;
  }
  else if(action==GetAction)
  {
    eigen_internal_assert(v!=0);
    #ifdef EIGEN_HAS_OPENMP
    if(m_maxThreads>0)
      *v = m_maxThreads;
    else
      *v = omp_get_max_threads();
    #else
    *v = 1;
    #endif
  }
  else
  {
    eigen_internal_assert(false);
  }
}

}

/** Must be call first when calling Eigen from multiple threads */
inline void initParallel()
{
  int nbt;
  internal::manage_multi_threading(GetAction, &nbt);
  std::ptrdiff_t l1, l2, l3;
  internal::manage_caching_sizes(GetAction, &l1, &l2, &l3);
}

/** \returns the max number of threads reserved for Eigen
  * \sa setNbThreads */
inline int nbThreads()
{
  int ret;
  internal::manage_multi_threading(GetAction, &ret);
  return ret;
}

/** Sets the max number of threads reserved for Eigen
  * \sa nbThreads */
inline void setNbThreads(int v)
{
  internal::manage_multi_threading(SetAction, &v);
}

namespace internal {

template<typename Index> struct GemmParallelInfo
{
  GemmParallelInfo() : sync(-1), users(0), lhs_start(0), lhs_length(0) {}

  int volatile sync;
  int volatile users;

  Index lhs_start;
  Index lhs_length;
};

template<bool Condition, typename Functor, typename Index>
void parallelize_gemm(const Functor& func, Index rows, Index cols, bool transpose)
{
  // TODO when EIGEN_USE_BLAS is defined,
  // we should still enable OMP for other scalar types
#if !(defined (EIGEN_HAS_OPENMP)) || defined (EIGEN_USE_BLAS)
  // FIXME the transpose variable is only needed to properly split
  // the matrix product when multithreading is enabled. This is a temporary
  // fix to support row-major destination matrices. This whole
  // parallelizer mechanism has to be redisigned anyway.
  EIGEN_UNUSED_VARIABLE(transpose);
  func(0,rows, 0,cols);
#else

  // Dynamically check whether we should enable or disable OpenMP.
  // The conditions are:
  // - the max number of threads we can create is greater than 1
  // - we are not already in a parallel code
  // - the sizes are large enough

  // 1- are we already in a parallel session?
  // FIXME omp_get_num_threads()>1 only works for openmp, what if the user does not use openmp?
  if((!Condition) || (omp_get_num_threads()>1))
    return func(0,rows, 0,cols);

  Index size = transpose ? rows : cols;

  // 2- compute the maximal number of threads from the size of the product:
  // FIXME this has to be fine tuned
  Index max_threads = std::max<Index>(1,size / 32);

  // 3 - compute the number of threads we are going to use
  Index threads = std::min<Index>(nbThreads(), max_threads);

  if(threads==1)
    return func(0,rows, 0,cols);

  Eigen::initParallel();
  func.initParallelSession();

  if(transpose)
    std::swap(rows,cols);

  Index blockCols = (cols / threads) & ~Index(0x3);
  Index blockRows = (rows / threads);
  blockRows = (blockRows/Functor::Traits::mr)*Functor::Traits::mr;
  
  GemmParallelInfo<Index>* info = new GemmParallelInfo<Index>[threads];

  #pragma omp parallel num_threads(threads)
  {
    Index i = omp_get_thread_num();
    Index r0 = i*blockRows;
    Index actualBlockRows = (i+1==threads) ? rows-r0 : blockRows;

    Index c0 = i*blockCols;
    Index actualBlockCols = (i+1==threads) ? cols-c0 : blockCols;

    info[i].lhs_start = r0;
    info[i].lhs_length = actualBlockRows;

    if(transpose) func(c0, actualBlockCols, 0, rows, info);
    else          func(0, rows, c0, actualBlockCols, info);
  }

  delete[] info;
#endif
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PARALLELIZER_H
