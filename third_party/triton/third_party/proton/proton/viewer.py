import argparse
from collections import namedtuple
import json
import pandas as pd
try:
    import hatchet as ht
    from hatchet.query import NegationQuery
except ImportError:
    raise ImportError("Failed to import hatchet. `pip install llnl-hatchet` to get the correct version.")
import numpy as np
from triton.profiler.hook import COMPUTE_METADATA_SCOPE_NAME, TritonHook


def match_available_metrics(metrics, inclusive_metrics, exclusive_metrics):
    ret = []
    if not isinstance(metrics, list):
        metrics = [metrics]
    if metrics:
        for metric in metrics:
            metric = metric.lower()
            for raw_metric in inclusive_metrics + exclusive_metrics:
                suffix = " (inc)" if raw_metric in inclusive_metrics else ""
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                if metric in (raw_metric, raw_metric_no_unit):
                    ret.append(raw_metric + suffix)
                    break
    if len(ret) == 0:
        raise RuntimeError(f"Metric {metric} is not found. Use the --list flag to list available metrics")
    return ret


def remove_frames(database: json):
    # We first fine frames that match either one of the two conditions:
    # 1. The frame name is COMPUTE_METADATA_SCOPE_NAME
    # 2. The frame has no metrics and no children
    # Then we go up from the located nodes and remove the parents if all children were
    # metadata nodes
    def remove_frame_helper(node):
        if "frame" not in node:
            return node
        if node["frame"]["name"] == COMPUTE_METADATA_SCOPE_NAME:
            return None
        if len(node["metrics"]) == 0 and len(node["children"]) == 0:
            return None
        children = node.get("children", [])
        new_children = []
        for child in children:
            new_child = remove_frame_helper(child)
            if new_child is not None:
                new_children.append(new_child)
        if len(new_children) > 0 or len(children) == 0:
            node["children"] = new_children
            return node
        return None

    new_database = []
    for node in database:
        new_node = remove_frame_helper(node)
        if new_node is not None:
            new_database.append(new_node)
    return new_database


def get_raw_metrics(file):
    database = json.load(file)
    database = remove_frames(database)
    device_info = database.pop(1)
    gf = ht.GraphFrame.from_literal(database)
    inclusive_metrics = gf.show_metric_columns()
    exclusive_metrics = [metric for metric in gf.dataframe.columns if metric not in inclusive_metrics]
    return gf, inclusive_metrics, exclusive_metrics, device_info


def get_min_time_flops(df, device_info):
    min_time_flops = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            arch = device_info[device_type][device_index]["arch"]
            num_sms = device_info[device_type][device_index]["num_sms"]
            clock_rate = device_info[device_type][device_index]["clock_rate"]
            for width in TritonHook.flops_width:
                idx = df["device_id"] == device_index
                device_frames = df[idx]
                if f"flops{width}" not in device_frames.columns:
                    continue
                max_flops = 0
                if device_type == "CUDA":
                    if arch == "80":
                        max_flops = 624e12 / (width / 8)
                    elif arch == "89":
                        # TODO(Keren): Implement fp16 acc-> 660.6 fp8
                        max_flops = (330.3 * 1e12) / (width / 8)
                    elif arch == "90":
                        # 114 sms and 1755mhz is the base number of sms and clock rate of H100 pcie
                        max_flops = ((num_sms / 114 * clock_rate / (1755 * 1e3) * 1513) * 1e12) / (width / 8)
                    elif arch == "100":
                        max_flops = (num_sms * 16384 * (clock_rate / 1e3) * 1e6) / (width / 8)
                elif device_type == "HIP":
                    if arch == "gfx90a":
                        max_flops = 383e12 / (width / 8)
                    elif arch == "gfx941" or arch == "gfx942":
                        max_flops = 2614.9e12 / (width / 8)
                else:
                    raise ValueError(f"Unsupported device type: {device_type}")
                min_time_flops.loc[idx, "min_time"] += device_frames[f"flops{width}"].fillna(0) / max_flops
    return min_time_flops


def get_min_time_bytes(df, device_info):
    min_time_bytes = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            idx = df["device_id"] == device_index
            device_frames = df[idx]
            memory_clock_rate = device_info[device_type][device_index]["memory_clock_rate"]  # in khz
            bus_width = device_info[device_type][device_index]["bus_width"]  # in bits
            peak_bandwidth = 2 * bus_width * memory_clock_rate * 1e3 / 8
            min_time_bytes.loc[idx, "min_time"] += device_frames["bytes"] / peak_bandwidth
    return min_time_bytes


FactorDict = namedtuple("FactorDict", ["name", "factor"])
time_factor_dict = FactorDict("time", {"time/s": 1, "time/ms": 1e-3, "time/us": 1e-6, "time/ns": 1e-9})
avg_time_factor_dict = FactorDict("avg_time", {f"avg_{key}": value for key, value in time_factor_dict.factor.items()})
cpu_time_factor_dict = FactorDict("cpu_time",
                                  {"cpu_time/s": 1, "cpu_time/ms": 1e-3, "cpu_time/us": 1e-6, "cpu_time/ns": 1e-9})
avg_cpu_time_factor_dict = FactorDict("avg_cpu_time",
                                      {f"avg_{key}": value
                                       for key, value in cpu_time_factor_dict.factor.items()})
bytes_factor_dict = FactorDict("bytes", {"byte/s": 1, "gbyte/s": 1e9, "tbyte/s": 1e12})

derivable_metrics = {
    **{key: bytes_factor_dict
       for key in bytes_factor_dict.factor.keys()},
}

# FLOPS have a specific width to their metric
default_flop_factor_dict = {"flop/s": 1, "gflop/s": 1e9, "tflop/s": 1e12}
derivable_metrics.update(
    {key: FactorDict("flops", default_flop_factor_dict)
     for key in default_flop_factor_dict.keys()})
for width in TritonHook.flops_width:
    factor_name = f"flops{width}"
    factor_dict = {f"flop{width}/s": 1, f"gflop{width}/s": 1e9, f"tflop{width}/s": 1e12}
    derivable_metrics.update({key: FactorDict(factor_name, factor_dict) for key in factor_dict.keys()})


def derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info):
    derived_metrics = []

    def get_time_seconds(df, metric, factor_dict):
        time_metric_name = match_available_metrics(metric, inclusive_metrics, exclusive_metrics)[0]
        time_unit = (factor_dict.name + "/" + time_metric_name.split("(")[1].split(")")[0])
        return df[time_metric_name] * factor_dict.factor[time_unit]

    for metric in metrics:
        if metric == "util":  # exclusive
            min_time_bytes = get_min_time_bytes(gf.dataframe, device_info)
            min_time_flops = get_min_time_flops(gf.dataframe, device_info)
            time_sec = get_time_seconds(gf.dataframe, "time", time_factor_dict)
            internal_frame_indices = gf.dataframe["device_id"].isna()
            gf.dataframe["util"] = min_time_flops["min_time"].combine(min_time_bytes["min_time"], max) / time_sec
            gf.dataframe.loc[internal_frame_indices, "util"] = np.nan
            derived_metrics.append("util")
        elif metric in derivable_metrics:  # flop<width>/s, <t/g>byte/s, inclusive
            derivable_metric = derivable_metrics[metric]
            metric_name = derivable_metric.name
            metric_factor_dict = derivable_metric.factor
            matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
            gf.dataframe[f"{metric} (inc)"] = (gf.dataframe[matched_metric_name] /
                                               (get_time_seconds(gf.dataframe, "time", time_factor_dict)) /
                                               metric_factor_dict[metric])
            derived_metrics.append(f"{metric} (inc)")
        elif metric in time_factor_dict.factor or metric in cpu_time_factor_dict.factor or \
                metric in avg_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor:  # inclusive
            is_cpu = metric in cpu_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor
            is_avg = metric in avg_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor

            factor_dict = (avg_cpu_time_factor_dict if is_avg else cpu_time_factor_dict) if is_cpu \
                else (avg_time_factor_dict if is_avg else time_factor_dict)
            metric_name = "cpu_time" if is_cpu else "time"
            metric_time_unit = factor_dict.name + "/" + metric.split("/")[1]

            time_value = get_time_seconds(gf.dataframe, metric_name, factor_dict)
            if is_avg:
                time_value = time_value / gf.dataframe["count (inc)"]

            gf.dataframe[f"{metric} (inc)"] = time_value / factor_dict.factor[metric_time_unit]
            derived_metrics.append(f"{metric} (inc)")
        else:
            metric_name_and_unit = metric.split("/")
            metric_name = metric_name_and_unit[0]
            if len(metric_name_and_unit) > 1:  # percentage, exclusive or inclusive
                metric_unit = metric_name_and_unit[1]
                if metric_unit != "%":
                    raise ValueError(f"Unsupported unit {metric_unit}")
                matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
                single_frame = gf.dataframe[matched_metric_name]
                suffix = ""
                if "(inc)" in matched_metric_name:
                    suffix = " (inc)"
                    total = gf.dataframe[matched_metric_name].iloc[0]
                else:
                    total = gf.dataframe[matched_metric_name].sum()
                gf.dataframe[metric + suffix] = (single_frame / total) * 100.0
                derived_metrics.append(metric + suffix)
            else:
                matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
                derived_metrics.append(matched_metric_name)

    # Update derived metrics to the graph frame
    for derived_metric in derived_metrics:
        if derived_metric.endswith("(inc)"):
            gf.inc_metrics.append(derived_metric)
        else:
            gf.exc_metrics.append(derived_metric)

    return derived_metrics


def format_frames(gf, format):
    if format == "file_function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split("/")[-1])
    elif format == "function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split(":")[-1])
    elif format == "file_function":
        gf.dataframe["name"] = gf.dataframe["name"].apply(
            lambda x: f"{x.split('/')[-1].split(':')[0]}@{x.split('@')[-1].split(':')[0]}")
    return gf


def filter_frames(gf, include=None, exclude=None, threshold=None, metric=None):
    if include:
        query = f"""
MATCH ("*")->(".", p)->("*")
WHERE p."name" =~ "{include}"
"""
        gf = gf.filter(query, squash=True)
    if exclude:
        inclusion_query = f"""
MATCH (".", p)->("*")
WHERE p."name" =~ "{exclude}"
"""
        query = NegationQuery(inclusion_query)
        gf = gf.filter(query, squash=True)
    if threshold:
        query = ["*", {metric: f">= {threshold}"}]
        gf = gf.filter(query, squash=True)
    return gf


def emit_warnings(gf, metrics):
    if "bytes (inc)" in metrics:
        byte_values = gf.dataframe["bytes (inc)"].values
        min_byte_value = np.nanmin(byte_values)
        if min_byte_value < 0:
            print("Warning: Negative byte values detected, this is usually the result of a datatype overflow\n")


def print_tree(gf, metrics, depth=100, format=None, print_sorted=False):
    gf = format_frames(gf, format)
    print(gf.tree(metric_column=metrics, expand_name=True, depth=depth, render_header=False))

    if print_sorted:
        print("Sorted kernels by metric " + metrics[0])
        sorted_df = gf.dataframe.sort_values(by=[metrics[0]], ascending=False)
        for row in range(1, len(sorted_df)):
            kernel_name = sorted_df.iloc[row]['name'][:100] + "..." if len(
                sorted_df.iloc[row]['name']) > 100 else sorted_df.iloc[row]['name']
            print("{:105} {:.4}".format(kernel_name, sorted_df.iloc[row][metrics[0]]))
    emit_warnings(gf, metrics)


def parse(metrics, filename, include=None, exclude=None, threshold=None):
    with open(filename, "r") as f:
        gf, inclusive_metrics, exclusive_metrics, device_info = get_raw_metrics(f)
        assert len(inclusive_metrics + exclusive_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        metrics = derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info)
        # TODO: generalize to support multiple metrics, not just the first one
        gf = filter_frames(gf, include, exclude, threshold, metrics[0])
        return gf, metrics


def show_metrics(file_name):
    with open(file_name, "r") as f:
        _, inclusive_metrics, exclusive_metrics, _ = get_raw_metrics(f)
        print("Available inclusive metrics:")
        if inclusive_metrics:
            for raw_metric in inclusive_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")
        print("Available exclusive metrics:")
        if exclusive_metrics:
            for raw_metric in exclusive_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")


def main():
    argparser = argparse.ArgumentParser(
        description="Performance data viewer for proton profiles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="""List available metrics. Metric names are case insensitive and ignore units.
Derived metrics can be created when source metrics are available.
- time/s, time/ms, time/us, time/ns: time
- avg_time/s, avg_time/ms, avg_time/us, avg_time/ns: time / count
- flop[<8/16/32/64>]/s, gflop[<8/16/32/64>]/s, tflop[<8/16/32/64>]/s: flops / time
- byte/s, gbyte/s, tbyte/s: bytes / time
- util: max(sum(flops<width>) / peak_flops<width>_time, sum(bytes) / peak_bandwidth_time)
- <metric>/%%: frame(metric) / sum(metric). Only availble for inclusive metrics (e.g. time)
""",
    )
    argparser.add_argument(
        "-m",
        "--metrics",
        type=str,
        default=None,
        help="""At maximum two metrics can be specified, separated by comma.
There are two modes:
1) Choose the output metric to display. It's case insensitive and ignore units.
2) Derive a new metric from existing metrics.
""",
    )
    argparser.add_argument(
        "-i",
        "--include",
        type=str,
        default=None,
        help=
        """Find frames that match the given regular expression and return all nodes in the paths that pass through the matching frames.
For example, the following command will display all paths that contain frames that contains "test":
```
proton-viewer -i ".*test.*" path/to/file.json
```
""",
    )
    argparser.add_argument(
        "-e",
        "--exclude",
        type=str,
        default=None,
        help="""Exclude frames that match the given regular expression and their children.
For example, the following command will exclude all paths starting from frames that contains "test":
```
proton-viewer -e ".*test.*" path/to/file.json
```
""",
    )
    argparser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help=
        "Exclude frames(kernels) whose metrics are below the given threshold. This filter only applies on the first metric.",
    )
    argparser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=100,
        help="The depth of the tree to display",
    )
    argparser.add_argument(
        "-f", "--format", type=str, choices=["full", "file_function_line", "function_line", "file_function"],
        default="full", help="""Formatting the frame name.
- full: include the path, file name, function name and line number.
- file_function_line: include the file name, function name and line number.
- function_line: include the function name and line number.
- file_function: include the file name and function name.
""")
    argparser.add_argument(
        "--print-sorted",
        action='store_true',
        default=False,
        help="Sort output by metric value instead of chronologically",
    )
    argparser.add_argument(
        "--diff-profile", "-diff", type=str, default=None,
        help="Compare two profiles. When used as 'proton-viewer -m time -diff file1.log file2.log', "
        "computes the difference: file2['time'] - file1['time']")

    args, target_args = argparser.parse_known_args()
    assert len(target_args) == 1, "Must specify a file to read"

    file_name = target_args[0]
    metrics = args.metrics.split(",") if args.metrics else None
    include = args.include
    exclude = args.exclude
    threshold = args.threshold
    depth = args.depth
    format = args.format
    diff = args.diff_profile
    print_sorted = args.print_sorted
    if include and exclude:
        raise ValueError("Cannot specify both include and exclude")
    if args.list:
        show_metrics(file_name)
    elif metrics:
        gf, derived_metrics = parse(metrics, file_name, include, exclude, threshold)
        if diff:
            gf2, _ = parse(metrics, diff, include, exclude, threshold)
            gf = gf.sub(gf2)
        print_tree(gf, derived_metrics, depth, format, print_sorted)


if __name__ == "__main__":
    main()
