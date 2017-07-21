import sys
import string
import re
from math import sqrt
from pprint import pprint
from pyspark.sql.functions import desc
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('correlate logs').getOrCreate()

assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.1'  # make sure we have Spark 2.1+

# regex that matches spaces and/or punctuation
wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]

    lines = spark.read.text(in_directory)
    words = lines.select(
        functions.explode(functions.split(
            lines['value'], wordbreak)).alias('words')
    )

    cleaned_words = words.select(
        functions.lower(words['words']).alias('words')
    )

    word_counts = cleaned_words.groupBy('words').agg(functions.count('*')) \
        .sort([desc('count(1)'), 'words'])\
        .filter(cleaned_words['words'] != '').coalesce(1)    # assingment indicates that the data aren't big enough to worry about

    word_counts.write.csv(out_directory, header=True, mode='overwrite')


if __name__ == '__main__':
    main()
