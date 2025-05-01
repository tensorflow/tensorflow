import tensorflow as tf

def parse_csv_line(csv_line, column_types):
    """
    Parses a single line of CSV data into a list of tensors based on the given column types.
    Supports int64, float32, and default (float32).
    
    Args:
    - csv_line: A list of strings representing the CSV data.
    - column_types: A list of types (e.g., tf.int64, tf.float32) for each column.

    Returns:
    - A list of tensors where each tensor corresponds to a parsed column value.
    """
    parsed_columns = []
    for col, dtype in zip(csv_line, column_types):
        if dtype == tf.int64:
            # Parse as int64
            parsed_columns.append(tf.strings.to_number(col, out_type=tf.int64))
        elif dtype == tf.float32:
            # Parse as float32
            parsed_columns.append(tf.strings.to_number(col, out_type=tf.float32))
        else:
            # Default to float32 for unsupported types
            parsed_columns.append(tf.strings.to_number(col, out_type=tf.float32))
    return parsed_columns

class CsvDataset(tf.data.Dataset):
    """
    A class for reading CSV data into a TensorFlow dataset. This class uses `parse_csv_line` to process
    and parse the CSV file rows based on the provided column types.
    """
    def __init__(self, filenames, column_types, buffer_size=1000, delimiter=","):
        """
        Args:
        - filenames: List of CSV filenames to read.
        - column_types: A list of TensorFlow data types (e.g., tf.int64, tf.float32).
        - buffer_size: Number of records to load at once (for batching).
        - delimiter: The delimiter separating columns in the CSV.
        """
        self.filenames = filenames
        self.column_types = column_types
        self.buffer_size = buffer_size
        self.delimiter = delimiter

    def _parse_function(self, csv_line):
        """
        This function is called for each CSV line to parse it into the correct types based on column_types.
        """
        csv_line = tf.strings.regex_replace(csv_line, self.delimiter, ",")  # Replace delimiter if needed
        csv_line = tf.strings.split(csv_line, self.delimiter)  # Split CSV line into columns
        return parse_csv_line(csv_line, self.column_types)

    def _generator(self):
        """
        Generator function to load and process each CSV file row by row.
        """
        for filename in self.filenames:
            # Open the file and process each line
            with tf.io.gfile.GFile(filename, 'r') as file:
                for line in file:
                    yield self._parse_function(line)

    def as_dataset(self):
        """
        Converts the generator to a TensorFlow Dataset.
        """
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=tf.TensorSpec(shape=(len(self.column_types),), dtype=tf.float32)
        )

# Example usage
if __name__ == "__main__":
    # Example CSV file with one int64 column and one float32 column
    filenames = ["sample.csv"]
    column_types = [tf.int64, tf.float32]  # First column is int64, second is float32

    dataset = CsvDataset(filenames=filenames, column_types=column_types)
    
    for parsed_row in dataset.as_dataset():
        print(parsed_row)
