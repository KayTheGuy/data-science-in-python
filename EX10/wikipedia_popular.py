import sys
from os.path import split
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()

assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.1'  # make sure we have Spark 2.1+


schema = types.StructType([
    types.StructField('lang', types.StringType(), False),
    types.StructField('title', types.StringType(), False),
    types.StructField('views', types.LongType(), False),
    types.StructField('bytes', types.LongType(), False),
])


def get_day_hour(path):
    ''' 
        extract hour in the format YYYYMMDD-HH from file path
    '''
    _, filename = split(path)
    hr = filename[11:22]
    return hr


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    get_day_hour_udf = functions.udf(
        get_day_hour, returnType=types.StringType())

    data = spark.read.csv(in_directory, sep=' ', schema=schema).withColumn(
        'hour', get_day_hour_udf(functions.input_file_name()))

    cleaned_data = data.filter(
        (data['lang'] == 'en') &
        (data['title'] != 'Main_Page') &
        (~ data['title'].startswith('Special:'))
    ).select(
        data['hour'],
        data['title'],
        data['views'],
    )
    groups = cleaned_data.groupBy('hour')
    grouped_data = groups.agg(
        functions.max(cleaned_data['views']).alias('views')
    ).join(cleaned_data, on='views').drop(cleaned_data['hour'])

    grouped_data = grouped_data.select(
        grouped_data['hour'],
        grouped_data['title'],
        grouped_data['views'],
    )
    result = grouped_data.sort(grouped_data['hour']).dropDuplicates().coalesce(2)
    result.write.csv(out_directory, mode='overwrite')

if __name__ == '__main__':
    main()
