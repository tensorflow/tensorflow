// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@googlemail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_STDDEQUE_H
#define EIGEN_STDDEQUE_H

#include "Eigen/src/StlSupport/details.h"

// Define the explicit instantiation (e.g. necessary for the Intel compiler)
#if defined(__INTEL_COMPILER) || defined(__GNUC__)
  #define EIGEN_EXPLICIT_STL_DEQUE_INSTANTIATION(...) template class std::deque<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> >;
#else
  #define EIGEN_EXPLICIT_STL_DEQUE_INSTANTIATION(...)
#endif

/**
 * This section contains a convenience MACRO which allows an easy specialization of
 * std::deque such that for data types with alignment issues the correct allocator
 * is used automatically.
 */
#define EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(...) \
EIGEN_EXPLICIT_STL_DEQUE_INSTANTIATION(__VA_ARGS__) \
namespace std \
{ \
  template<typename _Ay> \
  class deque<__VA_ARGS__, _Ay>  \
    : public deque<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > \
  { \
    typedef deque<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > deque_base; \
  public: \
    typedef __VA_ARGS__ value_type; \
    typedef typename deque_base::allocator_type allocator_type; \
    typedef typename deque_base::size_type size_type;  \
    typedef typename deque_base::iterator iterator;  \
    explicit deque(const allocator_type& a = allocator_type()) : deque_base(a) {}  \
    template<typename InputIterator> \
    deque(InputIterator first, InputIterator last, const allocator_type& a = allocator_type()) : deque_base(first, last, a) {} \
    deque(const deque& c) : deque_base(c) {}  \
    explicit deque(size_type num, const value_type& val = value_type()) : deque_base(num, val) {} \
    deque(iterator start, iterator end) : deque_base(start, end) {}  \
    deque& operator=(const deque& x) {  \
      deque_base::operator=(x);  \
      return *this;  \
    } \
  }; \
}

// check whether we really need the std::deque specialization
#if !(defined(_GLIBCXX_DEQUE) && (!EIGEN_GNUC_AT_LEAST(4,1))) /* Note that before gcc-4.1 we already have: std::deque::resize(size_type,const T&). */

namespace std {

#define EIGEN_STD_DEQUE_SPECIALIZATION_BODY \
  public:  \
    typedef T value_type; \
    typedef typename deque_base::allocator_type allocator_type; \
    typedef typename deque_base::size_type size_type;  \
    typedef typename deque_base::iterator iterator;  \
    typedef typename deque_base::const_iterator const_iterator;  \
    explicit deque(const allocator_type& a = allocator_type()) : deque_base(a) {}  \
    template<typename InputIterator> \
    deque(InputIterator first, InputIterator last, const allocator_type& a = allocator_type()) \
    : deque_base(first, last, a) {} \
    deque(const deque& c) : deque_base(c) {}  \
    explicit deque(size_type num, const value_type& val = value_type()) : deque_base(num, val) {} \
    deque(iterator start, iterator end) : deque_base(start, end) {}  \
    deque& operator=(const deque& x) {  \
      deque_base::operator=(x);  \
      return *this;  \
    }

  template<typename T>
  class deque<T,EIGEN_ALIGNED_ALLOCATOR<T> >
    : public deque<EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T),
                   Eigen::aligned_allocator_indirection<EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T)> >
{
  typedef deque<EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T),
                Eigen::aligned_allocator_indirection<EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T)> > deque_base;
  EIGEN_STD_DEQUE_SPECIALIZATION_BODY

  void resize(size_type new_size)
  { resize(new_size, T()); }

#if defined(_DEQUE_)
  // workaround MSVC std::deque implementation
  void resize(size_type new_size, const value_type& x)
  {
    if (deque_base::size() < new_size)
      deque_base::_Insert_n(deque_base::end(), new_size - deque_base::size(), x);
    else if (new_size < deque_base::size())
      deque_base::erase(deque_base::begin() + new_size, deque_base::end());
  }
  void push_back(const value_type& x)
  { deque_base::push_back(x); } 
  void push_front(const value_type& x)
  { deque_base::push_front(x); }
  using deque_base::insert;  
  iterator insert(const_iterator position, const value_type& x)
  { return deque_base::insert(position,x); }
  void insert(const_iterator position, size_type new_size, const value_type& x)
  { deque_base::insert(position, new_size, x); }
#elif defined(_GLIBCXX_DEQUE) && EIGEN_GNUC_AT_LEAST(4,2)
  // workaround GCC std::deque implementation
  void resize(size_type new_size, const value_type& x)
  {
    if (new_size < deque_base::size())
      deque_base::_M_erase_at_end(this->_M_impl._M_start + new_size);
    else
      deque_base::insert(deque_base::end(), new_size - deque_base::size(), x);
  }
#else
  // either GCC 4.1 or non-GCC
  // default implementation which should always work.
  void resize(size_type new_size, const value_type& x)
  {
    if (new_size < deque_base::size())
      deque_base::erase(deque_base::begin() + new_size, deque_base::end());
    else if (new_size > deque_base::size())
      deque_base::insert(deque_base::end(), new_size - deque_base::size(), x);
  }
#endif
  };
}

#endif // check whether specialization is actually required

#endif // EIGEN_STDDEQUE_H
