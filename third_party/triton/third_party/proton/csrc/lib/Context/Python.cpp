#include "Context/Python.h"
#include "pybind11/pybind11.h"
#include <algorithm>
#include <string>

namespace proton {

namespace {

// bpo-42262 added Py_NewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_NewRef)
PyObject *_Py_NewRef(PyObject *obj) {
  Py_INCREF(obj);
  return obj;
}
#define Py_NewRef(obj) _Py_NewRef((PyObject *)(obj))
#endif

// bpo-42262 added Py_XNewRef() to Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_XNewRef)
PyObject *_Py_XNewRef(PyObject *obj) {
  Py_XINCREF(obj);
  return obj;
}
#define Py_XNewRef(obj) _Py_XNewRef((PyObject *)(obj))
#endif

// bpo-40421 added PyFrame_GetCode() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1
PyCodeObject *getFrameCodeObject(PyFrameObject *frame) {
  assert(frame != nullptr);
  assert(frame->f_code != nullptr);
  return (PyCodeObject *)(Py_NewRef(frame->f_code));
}
#else
PyCodeObject *getFrameCodeObject(PyFrameObject *frame) {
  assert(frame != nullptr);
  return PyFrame_GetCode(frame);
}
#endif

// bpo-40421 added PyFrame_GetBack() to Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1
PyFrameObject *getFrameBack(PyFrameObject *frame) {
  assert(frame != nullptr);
  return (PyFrameObject *)(Py_XNewRef(frame->f_back));
}
#else
PyFrameObject *getFrameBack(PyFrameObject *frame) {
  assert(frame != nullptr);
  return PyFrame_GetBack(frame);
}
#endif

std::string unpackPyobject(PyObject *pyObject) {
  if (PyBytes_Check(pyObject)) {
    size_t size = PyBytes_GET_SIZE(pyObject);
    return std::string(PyBytes_AS_STRING(pyObject), size);
  }
  if (PyUnicode_Check(pyObject)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t size;
    const char *data = PyUnicode_AsUTF8AndSize(pyObject, &size);
    if (!data) {
      return "";
    }
    return std::string(data, (size_t)size);
  }
  return "";
}

} // namespace

std::vector<Context> PythonContextSource::getContextsImpl() {
  pybind11::gil_scoped_acquire gil;

  PyFrameObject *frame = PyEval_GetFrame();
  Py_XINCREF(frame);

  std::vector<Context> contexts;
  while (frame != nullptr) {
    PyCodeObject *f_code = getFrameCodeObject(frame);
    size_t lineno = PyFrame_GetLineNumber(frame);
    size_t firstLineNo = f_code->co_firstlineno;
    std::string file = unpackPyobject(f_code->co_filename);
    std::string function = unpackPyobject(f_code->co_name);
    auto pythonFrame = file + ":" + function + "@" + std::to_string(lineno);
    contexts.push_back(Context(pythonFrame));
    auto newFrame = getFrameBack(frame);
    Py_DECREF(frame);
    frame = newFrame;
  }
  std::reverse(contexts.begin(), contexts.end());
  return contexts;
}

size_t PythonContextSource::getDepth() { return getContextsImpl().size(); }

} // namespace proton
