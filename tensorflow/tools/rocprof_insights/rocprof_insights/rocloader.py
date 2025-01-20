"""Load and preprocess ROCm profiler data for multiple versions.

Classes:
    RocprofLoader: Loads and normalizes data from different rocprof versions.
"""

import pandas as pd

class RocprofLoader:
    """Loader class for ROCm profiler data from v1, v2, etc.

    This class reads data from a CSV file (or potentially other formats)
    and normalizes columns according to the version of rocprof.

    Attributes:
        filepath (str): Path to the data file.
        version (str): rocprof version (e.g., 'v1', 'v2', 'v3').
        df (pd.DataFrame): DataFrame containing loaded data.
    """

    VERSION_COLUMN_MAP = {
        'v1': {
            # Example placeholders; adjust based on actual columns:
        },
        'v2': {
            # Example placeholders; adjust based on actual columns:
        },
        'v3': {
            'Kernel_Name': 'kernel_name',
            'Start_Timestamp': 'start_ns',
            'End_Timestamp': 'end_ns',
            'Private_Segment_Size': 'local_memory',
            'Group_Segment_Size': 'lds',
            'DurationNs': 'duration_ns'
        },
        # If you have more versions, add them here
    }

    def __init__(self, filepath, version='v3'):
        """Initializes RocprofLoader.

        Args:
            filepath: Path to the data file (CSV).
            version: Version of the rocprof data (e.g., 'v1', 'v2').
        """
        self.filepath = filepath
        self.version = version.lower()
        self.df = None

    def load_data(self):
        """Loads the CSV data, renames columns, computes extra fields.

        Returns:
            pd.DataFrame: A pandas DataFrame with normalized columns.

        Raises:
            ValueError: If version is not supported or required columns are missing.
        """
        self.df = pd.read_csv(self.filepath)
        if self.version not in self.VERSION_COLUMN_MAP:
            raise ValueError(f"Unsupported rocprof version: {self.version}")

        # Rename columns based on the version
        rename_map = self.VERSION_COLUMN_MAP[self.version]
        self.df.rename(columns=rename_map, inplace=True, errors='ignore')

        # Ensure duration_ns is present; if not, try to compute it
        if 'duration_ns' not in self.df.columns:
            if 'start_ns' in self.df.columns and 'end_ns' in self.df.columns:
                self.df['duration_ns'] = self.df['end_ns'] - self.df['start_ns']
            else:
                raise ValueError("Cannot compute duration_ns (missing start_ns or end_ns).")

        # Convert to ms for convenience
        self.df['duration_ms'] = self.df['duration_ns'] / 1000.0

        return self.df