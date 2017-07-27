import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()

assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.1'      # make sure we have Spark 2.1+

schema = types.StructType([  
    types.StructField('author', types.StringType(), False),
    types.StructField('score', types.LongType(), False),
    types.StructField('subreddit', types.StringType(), False),
])


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]

    comments = spark.read.json(in_directory, schema=schema)

    avg_score = comments.groupBy('subreddit').agg(functions.avg(
        comments['score']))

    avg_positive_score = avg_score.filter(avg_score['avg(score)'] > 0)

    avg_positive_score = functions.broadcast(avg_positive_score)
    joined_scores = comments.join(avg_positive_score, on='subreddit')

    joined_relative_scores = joined_scores.select(
        joined_scores['subreddit'],
        joined_scores['author'],
        (joined_scores['score'] /
         joined_scores['avg(score)']).alias('rel_score')
    ).cache()

    max_rel_scores = joined_relative_scores.groupBy('subreddit').agg(
        functions.max(joined_relative_scores['rel_score'])).cache()

    max_rel_scores = functions.broadcast(max_rel_scores)
    best_author = joined_relative_scores.join(max_rel_scores, on=(
        max_rel_scores['max(rel_score)'] == joined_relative_scores['rel_score'])) \
        .drop(max_rel_scores['subreddit']).drop(max_rel_scores['max(rel_score)'])

    best_author = best_author.select(
        best_author['subreddit'],
        best_author['author'],
        best_author['rel_score'],
    )

    best_author.write.json(out_directory, mode='overwrite')


if __name__ == '__main__':
    main()
