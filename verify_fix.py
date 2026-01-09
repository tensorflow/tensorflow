
import sys
import os
import traceback

# Add the repository root to sys.path
sys.path.append(r'f:\New folder (2)')

try:
    print("Attempting to import tensorflow.python.eager.polymorphic_function.attributes...")
    from tensorflow.python.eager.polymorphic_function import attributes
    
    if hasattr(attributes, 'XLA_SEPARATE_COMPILED_GRADIENTS'):
        print("SUCCESS: XLA_SEPARATE_COMPILED_GRADIENTS exists.")
        print(f"Value: {attributes.XLA_SEPARATE_COMPILED_GRADIENTS}")
    else:
        print("FAILURE: XLA_SEPARATE_COMPILED_GRADIENTS not found.")
        
    if hasattr(attributes, 'XLA_SEPERATE_COMPILED_GRADIENTS'):
         print("FAILURE: XLA_SEPERATE_COMPILED_GRADIENTS still exists.")
    else:
        print("SUCCESS: XLA_SEPERATE_COMPILED_GRADIENTS (typo) correctly removed.")

    if attributes.XLA_SEPARATE_COMPILED_GRADIENTS in attributes.MONOMORPHIC_FUNCTION_ALLOWLIST:
        print("SUCCESS: XLA_SEPARATE_COMPILED_GRADIENTS is in MONOMORPHIC_FUNCTION_ALLOWLIST.")
    else:
        print("FAILURE: XLA_SEPARATE_COMPILED_GRADIENTS is NOT in MONOMORPHIC_FUNCTION_ALLOWLIST.")

except Exception:
    print("Error during verification:")
    traceback.print_exc()
