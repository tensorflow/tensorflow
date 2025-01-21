"""Visualization module using Plotly for rocprof data.

Classes:
    RocprofVisualizer: Provides methods to create histograms, pie charts,
                       and bar charts for kernels, APIs, and memcpy events.
"""

import plotly.express as px

class RocprofVisualizer:
    """A class containing methods to visualize ROCm profiler data with Plotly.

    Note:
        - For saving figures at high DPI, we rely on Plotly + kaleido.
        - Example usage:
            fig.write_image("output.png", scale=2)  # ~200 DPI
            fig.write_image("output.png", scale=4)  # ~400 DPI
    """

    def __init__(self, df):
        """Initializes RocprofVisualizer.

        Args:
            df (pd.DataFrame): DataFrame with raw or aggregated data.
        """
        self.df = df

    def histogram(
        self, 
        x_column, 
        color_column=None, 
        nbins=50, 
        title=None, 
        xaxis_label="Latency [ms]", 
        yaxis_label="Count [-]", 
        output_file=None
    ):
        """Creates and saves a histogram.

        Args:
            x_column (str): Numeric column to plot on the X-axis.
            color_column (str): Column to differentiate by color (optional).
            nbins (int): Number of histogram bins.
            title (str): Plot title.
            xaxis_label (str): Custom label for the X-axis.
            yaxis_label (str): Custom label for the Y-axis.
            output_file (str): If provided, saves the figure to file at ~400 DPI.
        """
        if x_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{x_column}' column.")

        fig = px.histogram(
            self.df,
            x=x_column,
            nbins=nbins,
            color=color_column,
            title=title or f"Histogram of {x_column}",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title=xaxis_label,
            yaxis_title=yaxis_label
        )

        # If saving to a static image, we can specify scale for higher DPI
        if output_file:
            fig.write_image(output_file, scale=4)  # ~400 DPI
        else:
            fig.show()

    def pie_chart(self, names_column, values_column, title=None, output_file=None):
        """Creates and saves a pie chart.

        Args:
            names_column (str): Column used for segment names.
            values_column (str): Column indicating numeric values for each segment.
            title (str): Plot title.
            output_file (str): If provided, saves the figure to file at ~400 DPI.
        """
        if names_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{names_column}' column.")
        if values_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{values_column}' column.")

        fig = px.pie(
            self.df,
            names=names_column,
            values=values_column,
            title=title or f"Pie Chart of {values_column} by {names_column}",
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(template="plotly_white")

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()

    def bar_chart(self, x_column, y_column, title=None, output_file=None):
        """Creates and saves a bar chart.

        Args:
            x_column (str): Column used for the X-axis (typically categorical).
            y_column (str): Column used for the Y-axis (numeric).
            title (str): Plot title.
            output_file (str): If provided, saves the figure to file at ~400 DPI.
        """
        if x_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{x_column}' column.")
        if y_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain '{y_column}' column.")
        
        fig = px.bar(
            self.df,
            x=x_column,
            y=y_column,
            title=title or f"Bar Chart of {y_column} by {x_column}",
        )
        fig.update_layout(template="plotly_white", xaxis={'categoryorder': 'total descending'})

        if output_file:
            fig.write_image(output_file, scale=4)
        else:
            fig.show()
