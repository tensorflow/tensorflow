# `class tensorflow::TensorShapeUtils`

Static helper routines for ` TensorShape `. Includes a few common predicates on a tensor shape.



###Member Details

#### `static bool tensorflow::TensorShapeUtils::IsScalar(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsScalar}





#### `static bool tensorflow::TensorShapeUtils::IsVector(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsVector}





#### `static bool tensorflow::TensorShapeUtils::IsVectorOrHigher(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsVectorOrHigher}





#### `static bool tensorflow::TensorShapeUtils::IsMatrix(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsMatrix}





#### `static bool tensorflow::TensorShapeUtils::IsSquareMatrix(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsSquareMatrix}





#### `static bool tensorflow::TensorShapeUtils::IsMatrixOrHigher(const TensorShape &shape)` {#static_bool_tensorflow_TensorShapeUtils_IsMatrixOrHigher}





#### `static Status tensorflow::TensorShapeUtils::MakeShape(const int32 *dims, int64 n, TensorShape *out)` {#static_Status_tensorflow_TensorShapeUtils_MakeShape}

Returns a ` TensorShape ` whose dimensions are `dims[0]`, `dims[1]`, ..., `dims[n-1]`.



#### `static Status tensorflow::TensorShapeUtils::MakeShape(const int64 *dims, int64 n, TensorShape *out)` {#static_Status_tensorflow_TensorShapeUtils_MakeShape}





#### `static Status tensorflow::TensorShapeUtils::MakeShape(gtl::ArraySlice< int32 > shape, TensorShape *out)` {#static_Status_tensorflow_TensorShapeUtils_MakeShape}





#### `static Status tensorflow::TensorShapeUtils::MakeShape(gtl::ArraySlice< int64 > shape, TensorShape *out)` {#static_Status_tensorflow_TensorShapeUtils_MakeShape}





#### `string tensorflow::TensorShapeUtils::ShapeListString(const gtl::ArraySlice< TensorShape > &shapes)` {#string_tensorflow_TensorShapeUtils_ShapeListString}





#### `bool tensorflow::TensorShapeUtils::StartsWith(const TensorShape &shape0, const TensorShape &shape1)` {#bool_tensorflow_TensorShapeUtils_StartsWith}




