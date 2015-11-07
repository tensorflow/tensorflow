// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPRESSED_STORAGE_H
#define EIGEN_COMPRESSED_STORAGE_H

namespace Eigen {

namespace internal {

/** \internal
  * Stores a sparse set of values as a list of values and a list of indices.
  *
  */
template<typename _Scalar,typename _Index>
class CompressedStorage
{
  public:

    typedef _Scalar Scalar;
    typedef _Index Index;

  protected:

    typedef typename NumTraits<Scalar>::Real RealScalar;

  public:

    CompressedStorage()
      : m_values(0), m_indices(0), m_size(0), m_allocatedSize(0)
    {}

    CompressedStorage(size_t size)
      : m_values(0), m_indices(0), m_size(0), m_allocatedSize(0)
    {
      resize(size);
    }

    CompressedStorage(const CompressedStorage& other)
      : m_values(0), m_indices(0), m_size(0), m_allocatedSize(0)
    {
      *this = other;
    }

    CompressedStorage& operator=(const CompressedStorage& other)
    {
      resize(other.size());
      internal::smart_copy(other.m_values,  other.m_values  + m_size, m_values);
      internal::smart_copy(other.m_indices, other.m_indices + m_size, m_indices);
      return *this;
    }

    void swap(CompressedStorage& other)
    {
      std::swap(m_values, other.m_values);
      std::swap(m_indices, other.m_indices);
      std::swap(m_size, other.m_size);
      std::swap(m_allocatedSize, other.m_allocatedSize);
    }

    ~CompressedStorage()
    {
      delete[] m_values;
      delete[] m_indices;
    }

    void reserve(size_t size)
    {
      size_t newAllocatedSize = m_size + size;
      if (newAllocatedSize > m_allocatedSize)
        reallocate(newAllocatedSize);
    }

    void squeeze()
    {
      if (m_allocatedSize>m_size)
        reallocate(m_size);
    }

    void resize(size_t size, float reserveSizeFactor = 0)
    {
      if (m_allocatedSize<size)
        reallocate(size + size_t(reserveSizeFactor*size));
      m_size = size;
    }

    void append(const Scalar& v, Index i)
    {
      Index id = static_cast<Index>(m_size);
      resize(m_size+1, 1);
      m_values[id] = v;
      m_indices[id] = i;
    }

    inline size_t size() const { return m_size; }
    inline size_t allocatedSize() const { return m_allocatedSize; }
    inline void clear() { m_size = 0; }

    inline Scalar& value(size_t i) { return m_values[i]; }
    inline const Scalar& value(size_t i) const { return m_values[i]; }

    inline Index& index(size_t i) { return m_indices[i]; }
    inline const Index& index(size_t i) const { return m_indices[i]; }

    static CompressedStorage Map(Index* indices, Scalar* values, size_t size)
    {
      CompressedStorage res;
      res.m_indices = indices;
      res.m_values = values;
      res.m_allocatedSize = res.m_size = size;
      return res;
    }

    /** \returns the largest \c k such that for all \c j in [0,k) index[\c j]\<\a key */
    inline Index searchLowerIndex(Index key) const
    {
      return searchLowerIndex(0, m_size, key);
    }

    /** \returns the largest \c k in [start,end) such that for all \c j in [start,k) index[\c j]\<\a key */
    inline Index searchLowerIndex(size_t start, size_t end, Index key) const
    {
      while(end>start)
      {
        size_t mid = (end+start)>>1;
        if (m_indices[mid]<key)
          start = mid+1;
        else
          end = mid;
      }
      return static_cast<Index>(start);
    }

    /** \returns the stored value at index \a key
      * If the value does not exist, then the value \a defaultValue is returned without any insertion. */
    inline Scalar at(Index key, const Scalar& defaultValue = Scalar(0)) const
    {
      if (m_size==0)
        return defaultValue;
      else if (key==m_indices[m_size-1])
        return m_values[m_size-1];
      // ^^  optimization: let's first check if it is the last coefficient
      // (very common in high level algorithms)
      const size_t id = searchLowerIndex(0,m_size-1,key);
      return ((id<m_size) && (m_indices[id]==key)) ? m_values[id] : defaultValue;
    }

    /** Like at(), but the search is performed in the range [start,end) */
    inline Scalar atInRange(size_t start, size_t end, Index key, const Scalar& defaultValue = Scalar(0)) const
    {
      if (start>=end)
        return Scalar(0);
      else if (end>start && key==m_indices[end-1])
        return m_values[end-1];
      // ^^  optimization: let's first check if it is the last coefficient
      // (very common in high level algorithms)
      const size_t id = searchLowerIndex(start,end-1,key);
      return ((id<end) && (m_indices[id]==key)) ? m_values[id] : defaultValue;
    }

    /** \returns a reference to the value at index \a key
      * If the value does not exist, then the value \a defaultValue is inserted
      * such that the keys are sorted. */
    inline Scalar& atWithInsertion(Index key, const Scalar& defaultValue = Scalar(0))
    {
      size_t id = searchLowerIndex(0,m_size,key);
      if (id>=m_size || m_indices[id]!=key)
      {
        resize(m_size+1,1);
        for (size_t j=m_size-1; j>id; --j)
        {
          m_indices[j] = m_indices[j-1];
          m_values[j] = m_values[j-1];
        }
        m_indices[id] = key;
        m_values[id] = defaultValue;
      }
      return m_values[id];
    }

    void prune(const Scalar& reference, const RealScalar& epsilon = NumTraits<RealScalar>::dummy_precision())
    {
      size_t k = 0;
      size_t n = size();
      for (size_t i=0; i<n; ++i)
      {
        if (!internal::isMuchSmallerThan(value(i), reference, epsilon))
        {
          value(k) = value(i);
          index(k) = index(i);
          ++k;
        }
      }
      resize(k,0);
    }

  protected:

    inline void reallocate(size_t size)
    {
      Scalar* newValues  = new Scalar[size];
      Index* newIndices = new Index[size];
      size_t copySize = (std::min)(size, m_size);
      // copy
      if (copySize>0) {
        internal::smart_copy(m_values, m_values+copySize, newValues);
        internal::smart_copy(m_indices, m_indices+copySize, newIndices);
      }
      // delete old stuff
      delete[] m_values;
      delete[] m_indices;
      m_values = newValues;
      m_indices = newIndices;
      m_allocatedSize = size;
    }

  protected:
    Scalar* m_values;
    Index* m_indices;
    size_t m_size;
    size_t m_allocatedSize;

};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPRESSED_STORAGE_H
