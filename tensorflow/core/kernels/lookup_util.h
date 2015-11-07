#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_LOOKUP_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_LOOKUP_UTIL_H_

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/initializable_lookup_table.h"

namespace tensorflow {
namespace lookup {

// Gets the LookupTable stored in the ctx->resource_manager() with key
// passed by attribute with name input_name, returns null if the table
// doesn't exist.
Status GetLookupTable(const string& input_name, OpKernelContext* ctx,
                      LookupInterface** table);

// Gets the InitializableLookupTable stored in the
// ctx->resource_manager() with key passed by attribute with name
// input_name, returns null if the table doesn't exist.
Status GetInitializableLookupTable(const string& input_name,
                                   OpKernelContext* ctx,
                                   InitializableLookupTable** table);

// Verify that the given key_dtype and value_dtype matches the corresponding
// table's data types.
Status CheckTableDataTypes(const LookupInterface& table, DataType key_dtype,
                           DataType value_dtype, const string& table_name);
}  // namespace lookup
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_LOOKUP_UTIL_H_
