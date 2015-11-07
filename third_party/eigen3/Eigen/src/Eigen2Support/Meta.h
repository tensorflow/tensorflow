// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN2_META_H
#define EIGEN2_META_H

namespace Eigen { 

template<typename T>
struct ei_traits : internal::traits<T>
{};

struct ei_meta_true {  enum { ret = 1 }; };
struct ei_meta_false { enum { ret = 0 }; };

template<bool Condition, typename Then, typename Else>
struct ei_meta_if { typedef Then ret; };

template<typename Then, typename Else>
struct ei_meta_if <false, Then, Else> { typedef Else ret; };

template<typename T, typename U> struct ei_is_same_type { enum { ret = 0 }; };
template<typename T> struct ei_is_same_type<T,T> { enum { ret = 1 }; };

template<typename T> struct ei_unref { typedef T type; };
template<typename T> struct ei_unref<T&> { typedef T type; };

template<typename T> struct ei_unpointer { typedef T type; };
template<typename T> struct ei_unpointer<T*> { typedef T type; };
template<typename T> struct ei_unpointer<T*const> { typedef T type; };

template<typename T> struct ei_unconst { typedef T type; };
template<typename T> struct ei_unconst<const T> { typedef T type; };
template<typename T> struct ei_unconst<T const &> { typedef T & type; };
template<typename T> struct ei_unconst<T const *> { typedef T * type; };

template<typename T> struct ei_cleantype { typedef T type; };
template<typename T> struct ei_cleantype<const T>   { typedef typename ei_cleantype<T>::type type; };
template<typename T> struct ei_cleantype<const T&>  { typedef typename ei_cleantype<T>::type type; };
template<typename T> struct ei_cleantype<T&>        { typedef typename ei_cleantype<T>::type type; };
template<typename T> struct ei_cleantype<const T*>  { typedef typename ei_cleantype<T>::type type; };
template<typename T> struct ei_cleantype<T*>        { typedef typename ei_cleantype<T>::type type; };

/** \internal In short, it computes int(sqrt(\a Y)) with \a Y an integer.
  * Usage example: \code ei_meta_sqrt<1023>::ret \endcode
  */
template<int Y,
         int InfX = 0,
         int SupX = ((Y==1) ? 1 : Y/2),
         bool Done = ((SupX-InfX)<=1 ? true : ((SupX*SupX <= Y) && ((SupX+1)*(SupX+1) > Y))) >
                                // use ?: instead of || just to shut up a stupid gcc 4.3 warning
class ei_meta_sqrt
{
    enum {
      MidX = (InfX+SupX)/2,
      TakeInf = MidX*MidX > Y ? 1 : 0,
      NewInf = int(TakeInf) ? InfX : int(MidX),
      NewSup = int(TakeInf) ? int(MidX) : SupX
    };
  public:
    enum { ret = ei_meta_sqrt<Y,NewInf,NewSup>::ret };
};

template<int Y, int InfX, int SupX>
class ei_meta_sqrt<Y, InfX, SupX, true> { public:  enum { ret = (SupX*SupX <= Y) ? SupX : InfX }; };

} // end namespace Eigen

#endif // EIGEN2_META_H
