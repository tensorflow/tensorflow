/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * Copyright (c) 2016 Oracle and/or its affiliates. All rights reserved.
 *
 * Oracle and Java are registered trademarks of Oracle and/or its affiliates.
 * Other names may be trademarks of their respective owners.
 *
 * The contents of this file are subject to the terms of either the GNU
 * General Public License Version 2 only ("GPL") or the Common
 * Development and Distribution License("CDDL") (collectively, the
 * "License"). You may not use this file except in compliance with the
 * License. You can obtain a copy of the License at
 * http://www.netbeans.org/cddl-gplv2.html
 * or nbbuild/licenses/CDDL-GPL-2-CP. See the License for the
 * specific language governing permissions and limitations under the
 * License.  When distributing the software, include this License Header
 * Notice in each file and include the License file at
 * nbbuild/licenses/CDDL-GPL-2-CP.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the GPL Version 2 section of the License file that
 * accompanied this code. If applicable, add the following below the
 * License Header, with the fields enclosed by brackets [] replaced by
 * your own identifying information:
 * "Portions Copyrighted [year] [name of copyright owner]"
 *
 * If you wish your version of this file to be governed by only the CDDL
 * or only the GPL Version 2, indicate your decision by adding
 * "[Contributor] elects to include this software in this distribution
 * under the [CDDL or GPL Version 2] license." If you do not indicate a
 * single choice of license, a recipient has the option to distribute
 * your version of this file under either the CDDL, the GPL Version 2 or
 * to extend the choice of license to its licensees as provided above.
 * However, if you add GPL Version 2 code and therefore, elected the GPL
 * Version 2 license, then the option applies only if the new code is
 * made subject to such option by the copyright holder.
 *
 * Contributor(s):
 */

// List of standard headers was taken in http://en.cppreference.com/w/cpp/header

#include <cstdlib> 	    // General purpose utilities: program control, dynamic memory allocation, random numbers, sort and search
#include <csignal> 	    // Functions and macro constants for signal management
#include <csetjmp> 	    // Macro (and function) that saves (and jumps) to an execution context
#include <cstdarg> 	    // Handling of variable length argument lists
#include <typeinfo> 	    // Runtime type information utilities
#include <bitset> 	    // std::bitset class template
#include <functional> 	    // Function objects, designed for use with the standard algorithms
#include <utility> 	    // Various utility components
#include <ctime> 	    // C-style time/date utilites
#include <cstddef> 	    // typedefs for types such as size_t, NULL and others
#include <new> 	            // Low-level memory management utilities
#include <memory> 	    // Higher level memory management utilities
#include <climits>          // limits of integral types
#include <cfloat> 	    // limits of float types
#include <limits> 	    // standardized way to query properties of arithmetic types
#include <exception> 	    // Exception handling utilities
#include <stdexcept> 	    // Standard exception objects
#include <cassert> 	    // Conditionally compiled macro that compares its argument to zero
#include <cerrno>           // Macro containing the last error number
#include <cctype>           // functions to determine the type contained in character data
#include <cwctype>          // functions for determining the type of wide character data
#include <cstring> 	    // various narrow character string handling functions
#include <cwchar> 	    // various wide and multibyte string handling functions
#include <string> 	    // std::basic_string class template
#include <vector> 	    // std::vector container
#include <deque> 	    // std::deque container
#include <list> 	    // std::list container
#include <set> 	            // std::set and std::multiset associative containers
#include <map> 	            // std::map and std::multimap associative containers
#include <stack> 	    // std::stack container adaptor
#include <queue> 	    // std::queue and std::priority_queue container adaptors
#include <algorithm> 	    // Algorithms that operate on containers
#include <iterator> 	    // Container iterators
#include <cmath>            // Common mathematics functions
#include <complex>          // Complex number type
#include <valarray>         // Class for representing and manipulating arrays of values
#include <numeric>          // Numeric operations on values in containers
#include <iosfwd>           // forward declarations of all classes in the input/output library
#include <ios>              // std::ios_base class, std::basic_ios class template and several typedefs
#include <istream>          // std::basic_istream class template and several typedefs
#include <ostream>          // std::basic_ostream, std::basic_iostream class templates and several typedefs
#include <iostream>         // several standard stream objects
#include <fstream>          // std::basic_fstream, std::basic_ifstream, std::basic_ofstream class templates and several typedefs
#include <sstream>          // std::basic_stringstream, std::basic_istringstream, std::basic_ostringstream class templates and several typedefs
#include <strstream>        // std::strstream, std::istrstream, std::ostrstream(deprecated)
#include <iomanip>          // Helper functions to control the format or input and output
#include <streambuf>        // std::basic_streambuf class template
#include <cstdio>           // C-style input-output functions
#include <locale>           // Localization utilities
#include <clocale>          // C localization utilities
#include <ciso646>          // empty header. The macros that appear in iso646.h in C are keywords in C++
#if __cplusplus >= 201103L
#include <typeindex>        // (since C++11) 	std::type_index
#include <type_traits>      // (since C++11) 	Compile-time type information
#include <chrono>           // (since C++11) 	C++ time utilites
#include <initializer_list> // (since C++11) 	std::initializer_list class template
#include <tuple>            // (since C++11) 	std::tuple class template
#include <scoped_allocator> // (since C++11) 	Nested allocator class
#include <cstdint>          // (since C++11) 	fixed-size types and limits of other types
#include <cinttypes>        // (since C++11) 	formatting macros , intmax_t and uintmax_t math and conversions
#include <system_error>     // (since C++11) 	defines std::error_code, a platform-dependent error code
#include <cuchar>           // (since C++11) 	C-style Unicode character conversion functions
#include <array>            // (since C++11) 	std::array container
#include <forward_list>     // (since C++11) 	std::forward_list container
#include <unordered_set>    // (since C++11) 	std::unordered_set and std::unordered_multiset unordered associative containers
#include <unordered_map>    // (since C++11) 	std::unordered_map and std::unordered_multimap unordered associative containers
#include <random>           // (since C++11) 	Random number generators and distributions
#include <ratio>            // (since C++11) 	Compile-time rational arithmetic
#include <cfenv>            // (since C++11) 	Floating-point environment access functions
#include <codecvt>          // (since C++11) 	Unicode conversion facilities
#include <regex>            // (since C++11) 	Classes, algorithms and iterators to support regular expression processing
#include <atomic>           // (since C++11) 	Atomic operations library
#include <ccomplex>         // (since C++11)(deprecated in C++17) 	simply includes the header <complex>
#include <ctgmath>          // (since C++11)(deprecated in C++17) 	simply includes the headers <ccomplex> (until C++17)<complex> (since C++17) and <cmath>: the overloads equivalent to the contents of the C header tgmath.h are already provided by those headers
#include <cstdalign>        // (since C++11)(deprecated in C++17) 	defines one compatibility macro constant
#include <cstdbool>         // (since C++11)(deprecated in C++17) 	defines one compatibility macro constant
#include <thread>           // (since C++11) 	std::thread class and supporting functions
#include <mutex>            // (since C++11) 	mutual exclusion primitives
#include <future>           // (since C++11) 	primitives for asynchronous computations
#include <condition_variable> // (since C++11) 	thread waiting conditions
#endif
#if __cplusplus >= 201300L
#include <shared_mutex>     // (since C++14) 	shared mutual exclusion primitives
#endif
#if __cplusplus >= 201500L
#include <any>              // (since C++17) 	std::any class template
#include <optional>         // (since C++17) 	std::optional class template
#include <variant>          // (since C++17) 	std::variant class template
#include <memory_resource>  // (since C++17) 	Polymorphic allocators and memory resources
#include <string_view>      // (since C++17) 	std::basic_string_view class template
#include <execution>        // (since C++17) 	Predefined execution policies for parallel versions of the algorithms
#include <filesystem>       // (since C++17) 	std::path class and supporting functions
#endif
