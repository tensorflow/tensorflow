// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"fmt"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/editiondefaults"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/reflect/protoreflect"
)

var defaultsCache = make(map[Edition]EditionFeatures)

func init() {
	unmarshalEditionDefaults(editiondefaults.Defaults)
}

func unmarshalGoFeature(b []byte, parent EditionFeatures) EditionFeatures {
	for len(b) > 0 {
		num, _, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch num {
		case genid.GoFeatures_LegacyUnmarshalJsonEnum_field_number:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			parent.GenerateLegacyUnmarshalJSON = protowire.DecodeBool(v)
		default:
			panic(fmt.Sprintf("unkown field number %d while unmarshalling GoFeatures", num))
		}
	}
	return parent
}

func unmarshalFeatureSet(b []byte, parent EditionFeatures) EditionFeatures {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FeatureSet_FieldPresence_field_number:
				parent.IsFieldPresence = v == genid.FeatureSet_EXPLICIT_enum_value || v == genid.FeatureSet_LEGACY_REQUIRED_enum_value
				parent.IsLegacyRequired = v == genid.FeatureSet_LEGACY_REQUIRED_enum_value
			case genid.FeatureSet_EnumType_field_number:
				parent.IsOpenEnum = v == genid.FeatureSet_OPEN_enum_value
			case genid.FeatureSet_RepeatedFieldEncoding_field_number:
				parent.IsPacked = v == genid.FeatureSet_PACKED_enum_value
			case genid.FeatureSet_Utf8Validation_field_number:
				parent.IsUTF8Validated = v == genid.FeatureSet_VERIFY_enum_value
			case genid.FeatureSet_MessageEncoding_field_number:
				parent.IsDelimitedEncoded = v == genid.FeatureSet_DELIMITED_enum_value
			case genid.FeatureSet_JsonFormat_field_number:
				parent.IsJSONCompliant = v == genid.FeatureSet_ALLOW_enum_value
			default:
				panic(fmt.Sprintf("unkown field number %d while unmarshalling FeatureSet", num))
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.GoFeatures_LegacyUnmarshalJsonEnum_field_number:
				parent = unmarshalGoFeature(v, parent)
			}
		}
	}

	return parent
}

func featuresFromParentDesc(parentDesc protoreflect.Descriptor) EditionFeatures {
	var parentFS EditionFeatures
	switch p := parentDesc.(type) {
	case *File:
		parentFS = p.L1.EditionFeatures
	case *Message:
		parentFS = p.L1.EditionFeatures
	default:
		panic(fmt.Sprintf("unknown parent type %T", parentDesc))
	}
	return parentFS
}

func unmarshalEditionDefault(b []byte) {
	var ed Edition
	var fs EditionFeatures
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FeatureSetDefaults_FeatureSetEditionDefault_Edition_field_number:
				ed = Edition(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FeatureSetDefaults_FeatureSetEditionDefault_Features_field_number:
				fs = unmarshalFeatureSet(v, fs)
			}
		}
	}
	defaultsCache[ed] = fs
}

func unmarshalEditionDefaults(b []byte) {
	for len(b) > 0 {
		num, _, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch num {
		case genid.FeatureSetDefaults_Defaults_field_number:
			def, m := protowire.ConsumeBytes(b)
			b = b[m:]
			unmarshalEditionDefault(def)
		case genid.FeatureSetDefaults_MinimumEdition_field_number,
			genid.FeatureSetDefaults_MaximumEdition_field_number:
			// We don't care about the minimum and maximum editions. If the
			// edition we are looking for later on is not in the cache we know
			// it is outside of the range between minimum and maximum edition.
			_, m := protowire.ConsumeVarint(b)
			b = b[m:]
		default:
			panic(fmt.Sprintf("unkown field number %d while unmarshalling EditionDefault", num))
		}
	}
}

func getFeaturesFor(ed Edition) EditionFeatures {
	if def, ok := defaultsCache[ed]; ok {
		return def
	}
	panic(fmt.Sprintf("unsupported edition: %v", ed))
}
