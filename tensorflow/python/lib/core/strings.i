// Wrapper functions to provide a scripting-language-friendly interface
// to our string libraries.
//
// NOTE: as of 2005-01-13, this SWIG file is not used to generate a pywrap
//       library for manipulation of various string-related types or access
//       to the special string functions (Python has plenty). This SWIG file
//       should be %import'd so that other SWIG wrappers have proper access
//       to the types in //strings (such as the StringPiece object). We may
//       generate a pywrap at some point in the future.
//
// NOTE: (Dan Ardelean) as of 2005-11-15 added typemaps to convert Java String
//       arguments to C++ StringPiece& objects. This is required because a
//       StringPiece class does not make sense - the code SWIG generates for a
//       StringPiece class is useless, because it releases the buffer set in
//       StringPiece after creating the object. C++ StringPiece objects rely on
//       the buffer holding the data being allocated externally.

// NOTE: for now, we'll just start with what is needed, and add stuff
//       as it comes up.

%{
#include "tensorflow/core/lib/core/stringpiece.h"
%}

%typemap(typecheck) tensorflow::StringPiece = char *;
%typemap(typecheck) const tensorflow::StringPiece & = char *;

// "tensorflow::StringPiece" arguments can be provided by a simple Python 'str' string
// or a 'unicode' object. If 'unicode', it's translated using the default
// encoding, i.e., sys.getdefaultencoding(). If passed None, a tensorflow::StringPiece
// of zero length with a NULL pointer is provided.
%typemap(in) tensorflow::StringPiece {
  if ($input != Py_None) {
    char * buf;
    Py_ssize_t len;
%#if PY_VERSION_HEX >= 0x03030000
    /* Do unicode handling as PyBytes_AsStringAndSize doesn't in Python 3. */
    if (PyUnicode_Check($input)) {
      buf = PyUnicode_AsUTF8AndSize($input, &len);
      if (buf == NULL)
        SWIG_fail;
    } else {
%#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 3
%#  error "Unsupported Python 3.x C API version (3.3 or later required)."
%#endif
      if (PyBytes_AsStringAndSize($input, &buf, &len) == -1) {
        // Python has raised an error (likely TypeError or UnicodeEncodeError).
        SWIG_fail;
      }
%#if PY_VERSION_HEX >= 0x03030000
    }
%#endif
    $1.set(buf, len);
  }
}

// "const tensorflow::StringPiece&" arguments can be provided the same as
// "tensorflow::StringPiece", whose typemap is defined above.
%typemap(in) const tensorflow::StringPiece & (tensorflow::StringPiece temp) {
  if ($input != Py_None) {
    char * buf;
    Py_ssize_t len;
%#if PY_VERSION_HEX >= 0x03030000
    /* Do unicode handling as PyBytes_AsStringAndSize doesn't in Python 3. */
    if (PyUnicode_Check($input)) {
      buf = PyUnicode_AsUTF8AndSize($input, &len);
      if (buf == NULL)
        SWIG_fail;
    } else {
%#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 3
%#  error "Unsupported Python 3.x C API version (3.3 or later required)."
%#endif
      if (PyBytes_AsStringAndSize($input, &buf, &len) == -1) {
        // Python has raised an error (likely TypeError or UnicodeEncodeError).
        SWIG_fail;
      }
%#if PY_VERSION_HEX >= 0x03030000
    }
%#endif
    temp.set(buf, len);
  }
  $1 = &temp;
}

// C++ functions returning tensorflow::StringPiece will simply return bytes in Python,
// or None if the StringPiece contained a NULL pointer.
%typemap(out) tensorflow::StringPiece {
  if ($1.data()) {
    $result = PyString_FromStringAndSize($1.data(), $1.size());
  } else {
    Py_INCREF(Py_None);
    $result = Py_None;
  }
}
