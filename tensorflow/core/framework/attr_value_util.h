#ifndef TENSORFLOW_FRAMEWORK_ATTR_VALUE_UTIL_H_
#define TENSORFLOW_FRAMEWORK_ATTR_VALUE_UTIL_H_

#include <string>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// A human-readable rendering of attr_value, that is more concise than a
// text-format proto.
string SummarizeAttrValue(const AttrValue& attr_value);

// Generates an error if attr_value doesn't have the indicated attr type.
Status AttrValueHasType(const AttrValue& attr_value, StringPiece type);

// Converts a text proto value from "text" into the the field of *out
// indicated by "type" (e.g. from the type field of an AttrDef).
// Examples:
// * If type:"int" and text:"-14", then *out is set to "i: -14"
// * If type:"list(string)" and text:"['foo', 'bar']",
//   then *out is set to "list { s: ['foo', 'bar'] }"
// Returns true on success.
bool ParseAttrValue(StringPiece type, StringPiece text, AttrValue* out);

// Sets *out based on the type of value.
void SetAttrValue(const string& value, AttrValue* out);
void SetAttrValue(const char* value, AttrValue* out);
void SetAttrValue(StringPiece value, AttrValue* out);
void SetAttrValue(int64 value, AttrValue* out);
void SetAttrValue(int32 value, AttrValue* out);
void SetAttrValue(float value, AttrValue* out);
void SetAttrValue(double value, AttrValue* out);
void SetAttrValue(bool value, AttrValue* out);
void SetAttrValue(DataType value, AttrValue* out);
void SetAttrValue(const TensorShape& value, AttrValue* out);
void SetAttrValue(const Tensor& value, AttrValue* out);
void SetAttrValue(const TensorProto& value, AttrValue* out);
void SetAttrValue(const NameAttrList& value, AttrValue* out);

void SetAttrValue(gtl::ArraySlice<string> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<const char*> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<int64> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<int32> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<float> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<double> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<bool> value, AttrValue* out);
void SetAttrValue(const std::vector<bool>& value, AttrValue* out);
void SetAttrValue(std::initializer_list<bool> value, AttrValue* out);
void SetAttrValue(DataTypeSlice value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<TensorShape> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<Tensor> value, AttrValue* out);
void SetAttrValue(gtl::ArraySlice<TensorProto> value, AttrValue* out);

inline void SetAttrValue(const AttrValue& value, AttrValue* out) {
  *out = value;
}

// Returns true if a and b have the same value.
// NOTE: May return false negatives for tensor values.
bool AreAttrValuesEqual(const AttrValue& a, const AttrValue& b);

// Returns true if "val" has a placeholder.
bool HasPlaceHolder(const AttrValue& val);

// SubstitutePlaceholders recursively replaces placeholders in 'value'
// with an attr value by calling SubstituteFunc. Returns true iff all
// placeholders in "value" are replaced with a value.
//
// SubstituteFunc is given a placeholder string. If the placeholder is
// unknown, SubstituteFunc returns false. Otherwise, overwrites the
// attr value and returns true.
typedef std::function<bool(const string&, AttrValue*)> SubstituteFunc;
bool SubstitutePlaceholders(SubstituteFunc substitute, AttrValue* value);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_ATTR_VALUE_UTIL_H_
