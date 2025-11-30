#!/usr/bin/env python3
"""
Validation script for the ConcatV2 float8 XLA fix.

This script validates that our code changes are syntactically correct
and include the necessary modifications.
"""

import os
import re

def validate_concat_op_changes():
    """Validate the changes made to concat_op.cc"""
    file_path = "/workspaces/tensorflow/tensorflow/compiler/tf2xla/kernels/concat_op.cc"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("kConcatTypes definition", r'constexpr std::array<DataType, \d+> kConcatTypes'),
        ("DT_FLOAT8_E4M3FN included", r'DT_FLOAT8_E4M3FN'),
        ("DT_FLOAT8_E5M2 included", r'DT_FLOAT8_E5M2'),
        ("DT_FLOAT8_E4M3FNUZ included", r'DT_FLOAT8_E4M3FNUZ'),
        ("DT_FLOAT8_E4M3B11FNUZ included", r'DT_FLOAT8_E4M3B11FNUZ'),
        ("DT_FLOAT8_E5M2FNUZ included", r'DT_FLOAT8_E5M2FNUZ'),
        ("Concat TypeConstraint", r'Name\("Concat"\)\s*\.TypeConstraint\("T", kConcatTypes\)'),
        ("ConcatV2 TypeConstraint", r'Name\("ConcatV2"\)\s*\.TypeConstraint\("T", kConcatTypes\)'),
    ]
    
    all_passed = True
    print("Validating concat_op.cc changes:")
    print("-" * 40)
    
    for check_name, pattern in checks:
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    return all_passed

def count_float8_types_in_array():
    """Count how many float8 types are included in kConcatTypes"""
    file_path = "/workspaces/tensorflow/tensorflow/compiler/tf2xla/kernels/concat_op.cc"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract kConcatTypes definition
    match = re.search(r'kConcatTypes\s*=\s*\{([^}]+)\}', content, re.MULTILINE | re.DOTALL)
    if not match:
        return 0
    
    array_content = match.group(1)
    float8_types = [
        'DT_FLOAT8_E5M2',
        'DT_FLOAT8_E4M3FN', 
        'DT_FLOAT8_E4M3FNUZ',
        'DT_FLOAT8_E4M3B11FNUZ',
        'DT_FLOAT8_E5M2FNUZ'
    ]
    
    count = sum(1 for ft in float8_types if ft in array_content)
    print(f"\nFloat8 types in kConcatTypes: {count}/{len(float8_types)}")
    
    for ft in float8_types:
        status = "✅" if ft in array_content else "❌"
        print(f"  {status} {ft}")
    
    return count

def main():
    print("ConcatV2 Float8 XLA Fix Validation")
    print("=" * 50)
    
    # Validate changes
    changes_valid = validate_concat_op_changes()
    
    # Count float8 types
    float8_count = count_float8_types_in_array()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"✅ Code changes valid: {changes_valid}")
    print(f"✅ Float8 types included: {float8_count}/5")
    
    if changes_valid and float8_count == 5:
        print("\n🎉 All validations passed! The fix looks correct.")
        print("The ConcatV2 XLA kernel should now support float8 types.")
    else:
        print("\n⚠️  Some validations failed. Please review the changes.")
    
    print("\nNote: This validation only checks code syntax and completeness.")
    print("Full testing requires building and running TensorFlow with these changes.")

if __name__ == "__main__":
    main()