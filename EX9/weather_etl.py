import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()

assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.1'  # make sure we have Spark 2.1+

observation_schema = types.StructType([
    types.StructField('station', types.StringType(), False),
    types.StructField('date', types.StringType(), False),
    types.StructField('observation', types.StringType(), False),
    types.StructField('value', types.IntegerType(), False),
    types.StructField('mflag', types.StringType(), False),
    types.StructField('qflag', types.StringType(), False),
    types.StructField('sflag', types.StringType(), False),
    types.StructField('obstime', types.StringType(), False),
])


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]

    weather = spark.read.csv(in_directory, schema=observation_schema)

    # clean data where:
    #   field qflag (quality flag) is null
    #   the station starts with 'CA'
    #   the observation is 'TMAX'
    weather = weather.where(
        weather['qflag'].isNull() &
        weather['station'].startswith('CA') &
        weather['observation'].startswith('TMAX')
    )

    # divide the temperature by 10 so it's actually in °C
    # keep only the columns station, date, and tmax (which is the value after
    # dividing by 10)
    cleaned_data = weather.select(
        weather['station'],
        weather['date'],
        (weather['value'] / 10).alias('tmax'),
    )

    cleaned_data.write.json(
        out_directory, compression='gzip', mode='overwrite')


if __name__ == '__main__':
    main()
