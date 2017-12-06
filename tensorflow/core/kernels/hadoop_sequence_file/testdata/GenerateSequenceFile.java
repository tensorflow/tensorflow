// A small java file converting txt file to sequence file for unittesting.
import java.io.File;
import java.io.IOException;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Metadata;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;

public class GenerateSequenceFile {
  public static void main(String[] args) throws IOException {
    String uri = args[1];
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    Path path = new Path(uri);
    IntWritable key = new IntWritable();
    BytesWritable value = new BytesWritable();
    File infile = new File(args[0]);
    SequenceFile.Writer writer = null;
    try {
      TreeMap<Text, Text> meta = new TreeMap<Text, Text>();
      meta.put(new Text("abc"), new Text("def"));
      meta.put(new Text("bla"), new Text("blublu"));
      writer = SequenceFile.createWriter(conf, Writer.file(path),
          Writer.keyClass(key.getClass()),
          Writer.valueClass(value.getClass()),
          Writer.bufferSize(fs.getConf().getInt("io.file.buffer.size",4096)),
          Writer.replication(fs.getDefaultReplication()),
          Writer.compression(SequenceFile.CompressionType.NONE),
          Writer.metadata(new Metadata(meta)));
      int ctr = 0;
      for (String line : FileUtils.readLines(infile)) {
        key.set(ctr++);
        value.set(line.getBytes(), 0, line.length());
        writer.append(key, value);
      }
    } finally {
      IOUtils.closeStream(writer);
    }
  }
}
