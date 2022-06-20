from datetime import date
from google.protobuf import text_format
from pytablewriter import MarkdownTableWriter # https://pypi.org/project/pytablewriter/
import support_pb2 as Support


today = date.today()
today = today.strftime("%Y-%m-%d")

supported_ops = Support.OpSupports()


def parse_supported_ops():
  with open("./support.pbtxt", 'r') as f:
    return text_format.Parse(f.read(), supported_ops)


supported_ops = parse_supported_ops()
table_name = "TF ops bridges coverage list"
fieldnames = ["op_name", "old_bridge", "new_bridge"]
old_bridge = new_bridge = all_bridges = 0
num_ops = len(supported_ops.support)
value_matrix = []

for op in supported_ops.support:
  old_bridge_value = new_bridge_value = ":white_check_mark:"
  if op.supports_old_bridge:
    old_bridge += 1
    old_bridge_value = ":heavy_check_mark:"
  if op.supports_new_bridge:
    new_bridge_value = ":heavy_check_mark:"
    new_bridge += 1
  if (op.supports_old_bridge and op.supports_new_bridge):
    all_bridges += 1
  value_matrix.append([op.graph_op_name, old_bridge_value, new_bridge_value])

writer = MarkdownTableWriter()

writer_ops_stats = MarkdownTableWriter(
    table_name="TF ops bridge coverage stats (%s)" % today,
    headers=["TF Ops", "Old bridge", "New bridge", "Both bridges"],
    value_matrix=[
        [num_ops, 
         "%i (%.2f%%)" % (old_bridge, old_bridge / num_ops * 100),
         "%i (%.2f%%)" % (new_bridge, new_bridge / num_ops * 100),
         "%i (%.2f%%)" % (all_bridges, all_bridges / num_ops * 100)
        ]
    ],
    margin=1
)

writer_ops_list = MarkdownTableWriter(
    table_name=table_name,
    headers=fieldnames,
    value_matrix=value_matrix,
    margin=1
)

with open("tf_os_bridges_coverage.md", "w") as f:
  writer.stream = f
  writer.from_writer(writer_ops_stats)
  writer.write_table()
  writer.from_writer(writer_ops_list)
  writer.write_table()
