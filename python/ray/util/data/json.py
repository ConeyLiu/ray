import pyarrow as pa
from pyarrow import fs
import os
import io
import pandas as pd


data_path = "/Users/xianyang/datasets/t"


def gen_data():
    if os.path.exists(data_path):
        print("data already exists")
        return

    import pyspark
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.range(0, 10000, 1, 2)
    df = df.withColumn("x", df.id + 1)
    df = df.withColumn("y", df.id + 2)
    df.write.json(data_path)
    spark.stop()


gen_data()

local, _ = fs.FileSystem.from_uri(data_path)
infos = local.get_file_info(fs.FileSelector(data_path, recursive=True))
for info in infos:
    p = os.path.basename(info.path)
    if info.type == fs.FileType.File and not p.startswith(".") and not p.startswith("_"):
        print(info)
        input_file = local.open_input_file(info.path)
        stream = local._wrap_input_stream(input_file, info.path, compression='detect', buffer_size=None)
        stream = io.TextIOWrapper(io.BufferedReader(stream, buffer_size=1024))

        stream.seek(100)
        print(stream.tell())
        stream.readline()
        t = stream.tell()
        print(t)
        stream.seek(t)
        print(stream.readline())
