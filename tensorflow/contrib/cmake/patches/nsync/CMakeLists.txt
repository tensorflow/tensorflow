cmake_minimum_required (VERSION 2.8.12)

# nsync provides portable synchronization primitives, such as mutexes and
# condition variables.
project (nsync)

# Set variable NSYNC_LANGUAGE to "c++11" to build with C++11
# rather than C.

# Some builds need position-independent code.
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------
# Platform dependencies

# Many platforms use these posix related sources; even Win32.
set (NSYNC_POSIX_SRC
  "platform/posix/src/nsync_panic.c"
  "platform/posix/src/per_thread_waiter.c"
  "platform/posix/src/time_rep.c"
  "platform/posix/src/yield.c"
)

if (WIN32)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_definitions(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
  add_definitions(/wd8029)
endif()

# Many of the string matches below use a literal "X" suffix on both sides.
# This is because some versions of cmake treat (for example) "MSVC" (in quotes)
# as a reference to the variable MSVC, thus the expression
#      "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC"
# is false when ${CMAKE_C_COMPILER_ID} has the value "MSVC"!  See
#    https://cmake.org/cmake/help/v3.1/policy/CMP0054.html

# Pick the include directory for the operating system.
if ("${NSYNC_LANGUAGE}X" STREQUAL "c++11X")
  include_directories ("${PROJECT_SOURCE_DIR}/platform/c++11")
  add_definitions ("-DNSYNC_USE_CPP11_TIMEPOINT -DNSYNC_ATOMIC_CPP11")
  set (NSYNC_OS_CPP_SRC
    "platform/c++11/src/nsync_semaphore_mutex.cc"
    "platform/c++11/src/per_thread_waiter.cc"
    "platform/c++11/src/yield.cc"
    "platform/c++11/src/time_rep_timespec.cc"
    "platform/c++11/src/nsync_panic.cc"
  )
  if ("${CMAKE_SYSTEM_NAME}X" STREQUAL "WindowsX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/win32")
    add_compile_options ("/TP")
    set (NSYNC_OS_SRC
      "platform/win32/src/clock_gettime.c"
      "platform/win32/src/pthread_key_win32.cc"
      ${NSYNC_OS_CPP_SRC}
    )
    set (NSYNC_TEST_OS_SRC
      "platform/win32/src/start_thread.c"
    )
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "DarwinX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/macos")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
    # Some versions of MacOS, such as Sierra, require _DARWIN_C_SOURCE
    # when including certin C++ standard header files, such as <mutex>.
    add_definitions ("-D_DARWIN_C_SOURCE")
    add_compile_options ("-std=c++11")
    set (NSYNC_OS_SRC
      ${NSYNC_OS_CPP_SRC}
      "platform/posix/src/clock_gettime.c"
      "platform/posix/src/nsync_semaphore_mutex.c"
    )
    set (NSYNC_TEST_OS_SRC
      "platform/posix/src/start_thread.c"
    )
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "LinuxX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
    add_compile_options ("-std=c++11")
    set (NSYNC_OS_SRC
      ${NSYNC_OS_CPP_SRC}
    )
    set (NSYNC_TEST_OS_SRC
      "platform/posix/src/start_thread.c"
    )
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "NetBSDX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
    add_compile_options ("-std=c++11")
    set (NSYNC_OS_SRC
      ${NSYNC_OS_CPP_SRC}
    )
    set (NSYNC_TEST_OS_SRC
      "platform/posix/src/start_thread.c"
    )
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "FreeBSDX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
    add_compile_options ("-std=c++11")
    set (NSYNC_OS_SRC
      ${NSYNC_OS_CPP_SRC}
    )
    set (NSYNC_TEST_OS_SRC
      "platform/posix/src/start_thread.c"
    )
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "OpenBSDX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
    add_compile_options ("-std=c++11")
    set (NSYNC_OS_SRC
      ${NSYNC_OS_CPP_SRC}
    )
    set (NSYNC_TEST_OS_SRC
      "platform/posix/src/start_thread.c"
    )
  endif ()
endif ()

# Pick the include directory for the compiler.
if ("${CMAKE_C_COMPILER_ID}X" STREQUAL "GNUX")
  include_directories ("${PROJECT_SOURCE_DIR}/platform/gcc")
  set (THREADS_HAVE_PTHREAD_ARG ON)
elseif ("${CMAKE_C_COMPILER_ID}X" STREQUAL "ClangX")
  include_directories ("${PROJECT_SOURCE_DIR}/platform/clang")
  set (THREADS_HAVE_PTHREAD_ARG ON)
elseif ("${CMAKE_C_COMPILER_ID}X" STREQUAL "MSVCX")
  include_directories ("${PROJECT_SOURCE_DIR}/platform/msvc")
else ()
  message (WARNING "CMAKE_C_COMPILER_ID (${CMAKE_C_COMPILER_ID}) matched NOTHING")
endif ()

if (NOT "${NSYNC_LANGUAGE}X" STREQUAL "c++11X")
  if ("${CMAKE_SYSTEM_NAME}X" STREQUAL "WindowsX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/win32")
    set (NSYNC_OS_SRC
      ${NSYNC_POSIX_SRC}
      "platform/win32/src/clock_gettime.c"
      "platform/win32/src/init_callback_win32.c"
      "platform/win32/src/nanosleep.c"
      "platform/win32/src/nsync_semaphore_win32.c"
      "platform/win32/src/pthread_cond_timedwait_win32.c"
      "platform/win32/src/pthread_key_win32.cc"
    )
    set (NSYNC_TEST_OS_SRC
      "platform/win32/src/start_thread.c"
    )
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "DarwinX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/macos")
    set (NSYNC_POSIX ON)
    set (NSYNC_OS_EXTRA_SRC
      "platform/posix/src/clock_gettime.c"
      "platform/posix/src/nsync_semaphore_mutex.c"
    )
    include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "LinuxX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/linux")
    set (NSYNC_POSIX ON)
    set (NSYNC_OS_EXTRA_SRC
         "platform/linux/src/nsync_semaphore_futex.c"
    )
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "NetBSDX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/netbsd")
    set (NSYNC_POSIX ON)
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "FreeBSDX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/freebsd")
    set (NSYNC_POSIX ON)
  elseif ("${CMAKE_SYSTEM_NAME}X" STREQUAL "OpenBSDX")
    include_directories ("${PROJECT_SOURCE_DIR}/platform/openbsd")
    set (NSYNC_POSIX ON)
  endif ()
endif ()

if (NSYNC_POSIX)
  include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
  set (NSYNC_OS_SRC
    ${NSYNC_POSIX_SRC}
    ${NSYNC_OS_EXTRA_SRC}
  )
  set (NSYNC_TEST_OS_SRC
    "platform/posix/src/start_thread.c"
  )
endif ()

# Pick the include directory for the architecture.
if (("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "x86_64X") OR
    ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "amd64X") OR
    ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "AMD64X"))
  include_directories ("${PROJECT_SOURCE_DIR}/platform/x86_64")
elseif (("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "x86_32X") OR
  ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "i386X") OR
        ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "i686X"))
  include_directories ("${PROJECT_SOURCE_DIR}/platform/x86_32")
elseif (("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "armv6lX") OR
  ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "armv7lX") OR
  ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "armX"))
  include_directories ("${PROJECT_SOURCE_DIR}/platform/arm")
elseif (("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "aarch64X") OR
  ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "arm64X"))
  include_directories ("${PROJECT_SOURCE_DIR}/platform/aarch64")
elseif (("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "ppcX") OR
  ("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "ppc32X"))
  include_directories ("${PROJECT_SOURCE_DIR}/platform/ppc32")
elseif (("${CMAKE_SYSTEM_PROCESSOR}X" STREQUAL "ppc64X"))
  include_directories ("${PROJECT_SOURCE_DIR}/platform/ppc64")
endif ()

# Windows uses some include files from the posix directory also.
if ("${CMAKE_SYSTEM_NAME}X" STREQUAL "WindowsX")
  include_directories ("${PROJECT_SOURCE_DIR}/platform/posix")
endif ()

# -----------------------------------------------------------------

include_directories ("${PROJECT_SOURCE_DIR}/public")
include_directories ("${PROJECT_SOURCE_DIR}/internal")

set (NSYNC_SRC
  "internal/common.c"
  "internal/counter.c"
  "internal/cv.c"
  "internal/debug.c"
  "internal/dll.c"
  "internal/mu.c"
  "internal/mu_wait.c"
  "internal/note.c"
  "internal/once.c"
  "internal/sem_wait.c"
  "internal/time_internal.c"
  "internal/wait.c"
  ${NSYNC_OS_SRC}
)
add_library (nsync ${NSYNC_SRC})

set (NSYNC_TEST_SRC
  "testing/array.c"
  "testing/atm_log.c"
  "testing/closure.c"
  "testing/smprintf.c"
  "testing/testing.c"
  "testing/time_extra.c"
  ${NSYNC_TEST_OS_SRC}
)
add_library (nsync_test ${NSYNC_TEST_SRC})

set (NSYNC_TESTS
  "counter_test"
  "cv_mu_timeout_stress_test"
  "cv_test"
  "cv_wait_example_test"
  "dll_test"
  "mu_starvation_test"
  "mu_test"
  "mu_wait_example_test"
  "mu_wait_test"
  "note_test"
  "once_test"
  "pingpong_test"
  "wait_test"
)

if ("${NSYNC_LANGUAGE}X" STREQUAL "c++11X")
  foreach (s IN ITEMS ${NSYNC_SRC} ${NSYNC_TEST_SRC})
    SET_SOURCE_FILES_PROPERTIES ("${s}" PROPERTIES LANGUAGE CXX)
  endforeach (s)
  foreach (t IN ITEMS ${NSYNC_TESTS})
    SET_SOURCE_FILES_PROPERTIES ("testing/${t}.c" PROPERTIES LANGUAGE CXX)
  endforeach (t)
endif ()

enable_testing ()
foreach (t IN ITEMS ${NSYNC_TESTS})
  add_executable (${t} "testing/${t}.c")
endforeach (t)

find_package (Threads REQUIRED)
set (THREADS_PREFER_PTHREAD_FLAG ON)
foreach (t IN ITEMS "nsync" "nsync_test" ${NSYNC_TESTS})
  if (THREADS_HAVE_PTHREAD_ARG)
    target_compile_options (${t} PUBLIC "-pthread")
  endif ()
  if (CMAKE_THREAD_LIBS_INIT)
    target_link_libraries (${t} "${CMAKE_THREAD_LIBS_INIT}")
  endif ()
endforeach (t)

foreach (t IN ITEMS ${NSYNC_TESTS})
  target_link_libraries (${t} nsync_test nsync)
  add_test (NAME ${t} COMMAND ${t})
endforeach (t)

install (TARGETS nsync
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

set (NSYNC_INCLUDES
  "public/nsync.h"
  "public/nsync_atomic.h"
  "public/nsync_counter.h"
  "public/nsync_cpp.h"
  "public/nsync_cv.h"
  "public/nsync_debug.h"
  "public/nsync_mu.h"
  "public/nsync_mu_wait.h"
  "public/nsync_note.h"
  "public/nsync_once.h"
  "public/nsync_time.h"
  "public/nsync_time_internal.h"
  "public/nsync_waiter.h"
)

foreach (NSYNC_INCLUDE ${NSYNC_INCLUDES})
  install (FILES ${NSYNC_INCLUDE} DESTINATION include COMPONENT Development)
endforeach ()
