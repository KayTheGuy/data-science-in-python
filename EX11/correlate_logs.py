import re
import sys
from math import sqrt
from pprint import pprint
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('correlate logs').getOrCreate()

assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.1'  # make sure we have Spark 2.1+

line_re = re.compile(
    "^(\\S+) - - \\[\\S+ [+-]\\d+\\] \"[A-Z]+ \\S+ HTTP/\\d\\.\\d\" \\d+ (\\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. 
    Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(hostname=m.group(1), bytes=m.group(2))
    else:
        return None


def not_none(row):
    """
    Is this None? 
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    return log_lines.map(line_to_row).filter(not_none)


def main():
    in_directory = sys.argv[1]
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    groups = logs.groupBy('hostname')

    data_points = groups.agg(
                            functions.count(logs['hostname']).alias('req_count'),
                            functions.sum(logs['bytes']).alias('req_bytes')
                            ).cache()

    one_group = data_points.groupBy()

    six_sums = one_group.agg(
                            functions.count(data_points['req_count']).alias('n'),
                            functions.sum(data_points['req_count']).alias('x'),
                            functions.sum(data_points['req_bytes']).alias('y'),
                            functions.sum(data_points['req_bytes'] * data_points['req_count']).alias('xy'),
                            functions.sum(data_points['req_count'] * data_points['req_count']).alias('x2'),
                            functions.sum(data_points['req_bytes'] * data_points['req_bytes']).alias('y2')
                            ).first()     
                                              
    n = six_sums['n']
    x = six_sums['x']
    y = six_sums['y']
    x2 = six_sums['x2']
    y2 = six_sums['y2']
    xy = six_sums['xy']

    r = ((n * xy) - (x * y)) / \
        (sqrt((n * x2) - (x ** 2)) * sqrt((n * y2) - (y ** 2)))
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__ == '__main__':
    main()
