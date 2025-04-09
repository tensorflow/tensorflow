import sys
from tensorflow.compiler.mlir.lite.python._pywrap_converter_api import _mlir
from tensorflow.compiler.mlir.lite.python._pywrap_converter_api._mlir import ir
sys.modules["tensorflow.compiler.mlir.lite.python.mlir._mlir_libs._mlir"] = _mlir
sys.modules["tensorflow.compiler.mlir.lite.python.mlir._mlir_libs._mlir.ir"] = ir
import tensorflow.compiler.mlir.lite.python.mlir._mlir_libs