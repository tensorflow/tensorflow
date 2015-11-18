# Class `tensorflow::TensorShapeUtils`

Static helper routines for ` TensorShape `. Includes a few common predicates on a tensor shape.



##Member Summary

* [`static bool tensorflow::TensorShapeUtils::IsScalar(const TensorShape &shape)`](#static_bool_tensorflow_TensorShapeUtils_IsScalar)
* [`static bool tensorflow::TensorShapeUtils::IsVector(const TensorShape &shape)`](#static_bool_tensorflow_TensorShapeUtils_IsVector)
* [`static bool tensorflow::TensorShapeUtils::IsLegacyScalar(const TensorShape &shape)`](#static_bool_tensorflow_TensorShapeUtils_IsLegacyScalar)
* [`static bool tensorflow::TensorShapeUtils::IsLegacyVector(const TensorShape &shape)`](#static_bool_tensorflow_TensorShapeUtils_IsLegacyVector)
* [`static bool tensorflow::TensorShapeUtils::IsVectorOrHigher(const TensorShape &shape)`](#static_bool_tensorflow_TensorShapeUtils_IsVectorOrHigher)
* [`static bool tensorflow::TensorShapeUtils::IsMatrix(const TensorShape &shape)`](#static_bool_tensorflow_TensorShapeUtils_IsMatrix)
* [`static bool tensorflow::TensorShapeUtils::IsMatrixOrHigher(const TensorShape &shape)`](#static_bool_tensorflow_TensorShapeUtils_IsMatrixOrHigher)
* [`static TensorShape tensorflow::TensorShapeUtils::MakeShape(const T *dims, int n)`](#static_TensorShape_tensorflow_TensorShapeUtils_MakeShape)
  * Returns a ` TensorShape ` whose dimensions are `dims[0]`, `dims[1]`, ..., `dims[n-1]`.
* [`static string tensorflow::TensorShapeUtils::ShapeListString(const gtl::ArraySlice< TensorShape > &shapes)`](#static_string_tensorflow_TensorShapeUtils_ShapeListString)
* [`static bool tensorflow::TensorShapeUtils::StartsWith(const TensorShape &shape0, const TensorShape &shape1)`](#static_bool_tensorflow_TensorShapeUtils_StartsWith)

##Member Details

#### `static bool tensorflow::TensorShapeUtils::IsScalar(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsScalar}





#### `static bool tensorflow::TensorShapeUtils::IsVector(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsVector}





#### `static bool tensorflow::TensorShapeUtils::IsLegacyScalar(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsLegacyScalar}





#### `static bool tensorflow::TensorShapeUtils::IsLegacyVector(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsLegacyVector}





#### `static bool tensorflow::TensorShapeUtils::IsVectorOrHigher(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsVectorOrHigher}





#### `static bool tensorflow::TensorShapeUtils::IsMatrix(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsMatrix}





#### `static bool tensorflow::TensorShapeUtils::IsMatrixOrHigher(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsMatrixOrHigher}





#### `static TensorShape tensorflow::TensorShapeUtils::MakeShape(const T *dims, int n)` {#static_TensorShape_tensorflow_TensorShapeUtils_MakeShape}

Returns a ` TensorShape ` whose dimensions are `dims[0]`, `dims[1]`, ..., `dims[n-1]`.



#### `static string tensorflow::TensorShapeUtils::ShapeListString(const gtl::ArraySlice< TensorShape > &shapes)` {#static_string_tensorflow_TensorShapeUtils_ShapeListString}





#### `static bool tensorflow::TensorShapeUtils::StartsWith(const TensorShape &shape0, const TensorShape &shape1)` {#static_bool_tensorflow_TensorShapeUtils_StartsWith}




