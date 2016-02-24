# `class tensorflow::Status`





###Member Details

#### `tensorflow::Status::Status()` {#tensorflow_Status_Status}

Create a success status.



#### `tensorflow::Status::~Status()` {#tensorflow_Status_Status}





#### `tensorflow::Status::Status(tensorflow::error::Code code, tensorflow::StringPiece msg)` {#tensorflow_Status_Status}

Create a status with the specified error code and msg as a human-readable string containing more detailed information.



#### `tensorflow::Status::Status(const Status &s)` {#tensorflow_Status_Status}

Copy the specified status.



#### `void tensorflow::Status::operator=(const Status &s)` {#void_tensorflow_Status_operator_}





#### `bool tensorflow::Status::ok() const` {#bool_tensorflow_Status_ok}

Returns true iff the status indicates success.



#### `tensorflow::error::Code tensorflow::Status::code() const` {#tensorflow_error_Code_tensorflow_Status_code}





#### `const string& tensorflow::Status::error_message() const` {#const_string_tensorflow_Status_error_message}





#### `bool tensorflow::Status::operator==(const Status &x) const` {#bool_tensorflow_Status_operator_}





#### `bool tensorflow::Status::operator!=(const Status &x) const` {#bool_tensorflow_Status_operator_}





#### `void tensorflow::Status::Update(const Status &new_status)` {#void_tensorflow_Status_Update}

If ` ok() `, stores `new_status` into `*this`. If `!ok()`, preserves the current status, but may augment with additional information about `new_status`.

Convenient way of keeping track of the first error encountered. Instead of: `if (overall_status.ok()) overall_status = new_status` Use: `overall_status.Update(new_status);`

#### `string tensorflow::Status::ToString() const` {#string_tensorflow_Status_ToString}

Return a string representation of this status suitable for printing. Returns the string `"OK"` for success.



#### `return tensorflow::Status::OK()` {#return_tensorflow_Status_OK}




