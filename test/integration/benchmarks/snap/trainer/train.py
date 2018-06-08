import json
import os

import sys

import argparse
import logging
import math

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.INFO)

logger = logging.getLogger('user-script')  # initialize logging class
format = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)

logging.getLogger('tensorflow').addHandler(sh)

import sagemaker_containers

import tensorflow as tf

tf.logging.set_verbosity(logging.INFO)

from tensorflow.contrib.layers import sparse_column_with_integerized_feature
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.canned.linear import LinearClassifier
from tensorflow.python.feature_column.feature_column import (bucketized_column, crossed_column,
                                                             embedding_column,
                                                             numeric_column)
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io

FORMAT_CATEGORICAL_FEATURE_ID = 'categorical-feature-{}_id'
FORMAT_INT_FEATURE = 'int-feature-{}'
LINEAR = 'linear'
DEEP = 'deep'
CROSSES = 'crosses'
CROSS_HASH_BUCKET_SIZE = int(1e6)

DATASETS = ['kaggle', 'large']
KAGGLE, LARGE = DATASETS

NUM_EXAMPLES = 'num_examples'

L2_REGULARIZATION = 'l2_regularization'

KEY_FEATURE_COLUMN = 'example_id'
TARGET_FEATURE_COLUMN = 'clicked'

PIPELINE_CONFIG = {
    KAGGLE: {
        NUM_EXAMPLES:
                 45 * 1e6,
        L2_REGULARIZATION:
                 60,
        CROSSES: [(27, 31), (33, 37), (27, 29), (4, 6), (19, 36), (19, 22),
                  (19, 33), (6, 9), (10, 5), (19, 35, 36), (30, 36), (30, 11),
                  (20, 30), (19, 22, 28), (27, 31, 39), (1, 8), (11, 5),
                  (11, 7), (25, 2), (26, 27, 31), (38, 5), (19, 22, 11),
                  (37, 5), (24, 11), (13, 4), (19, 8), (27, 31, 33),
                  (17, 19, 36), (31, 3), (26, 5), (30, 12), (27, 31, 2),
                  (11, 9), (15, 34), (19, 26, 36), (27, 36), (30, 5), (23, 37),
                  (13, 3), (31, 6), (26, 8), (30, 33), (27, 36, 37), (1, 6),
                  (17, 30), (20, 23), (27, 31, 35), (26, 1), (26, 27, 36)]
    },
    LARGE:  {
        NUM_EXAMPLES:
                 4 * 1e9,
        L2_REGULARIZATION:
                 500,
        CROSSES: [(19, 12), (10, 12), (10, 11), (32, 12), (30, 1), (36, 39),
                  (13, 3), (26, 32), (15, 23), (10, 9), (20, 25), (16, 26, 32),
                  (11, 12), (30, 10), (15, 38), (10, 6), (39, 8), (39, 10),
                  (19, 28, 12), (15, 37), (26, 7), (11, 5), (14, 39, 8),
                  (11, 2), (12, 4), (28, 1), (26, 32, 11), (26, 10, 7),
                  (22, 30), (15, 24, 38), (20, 10, 12), (32, 9), (15, 8),
                  (32, 4), (26, 3), (29, 30), (22, 30, 39), (22, 30, 36, 39),
                  (22, 26), (20, 11), (4, 9), (26, 12), (12, 13), (32, 6),
                  (39, 11), (15, 26, 32)]
    }
}


def feature_columns(config, model_type, vocab_sizes, use_crosses):
    """Return the feature columns with their names and types."""
    result = []
    boundaries = [1.5 ** j - 0.51 for j in range(40)]

    for index in range(1, 14):
        column = bucketized_column(numeric_column(FORMAT_INT_FEATURE.format(index), dtype=tf.int64),
                                   boundaries)
        result.append(column)

    if model_type == LINEAR:
        for index in range(14, 40):
            column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
            column = sparse_column_with_integerized_feature(column_name, vocab_sizes[column_name],
                                                            combiner='sum')
            result.append(column)
        if use_crosses:
            for cross in config[CROSSES]:
                column = crossed_column(
                    [result[index - 1] for index in cross],
                    hash_bucket_size=CROSS_HASH_BUCKET_SIZE,
                    # hash_key=SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
                )
                result.append(column)
    elif model_type == DEEP:
        for index in range(14, 40):
            column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
            vocab_size = vocab_sizes[column_name]
            column = sparse_column_with_integerized_feature(column_name, vocab_size, combiner='sum')
            embedding_size = int(math.floor(6 * vocab_size ** 0.25))
            embedding = embedding_column(column, embedding_size, combiner='mean')
            result.append(embedding)

    return result


def get_vocab_sizes():
    """Read vocabulary sizes from the metadata."""
    return {FORMAT_CATEGORICAL_FEATURE_ID.format(index): int(10 * 1000) for index in range(14, 40)}


def estimator_fn(run_config, dataset, use_crosses, model_type, hidden_units):
    print("estimator_fn run_config: {}".format(run_config))

    columns = feature_columns(PIPELINE_CONFIG.get(dataset), model_type,
                              get_vocab_sizes(), use_crosses)

    logger.info('test')

    if model_type == 'linear':
        return LinearClassifier(
            config=run_config,
            feature_columns=columns,
        )
    elif model_type == DEEP:
        return DNNClassifier(
            hidden_units=hidden_units,
            feature_columns=columns,
            config=run_config
        )


def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def train_input_fn(training_dir, batch_size):
    transformed_metadata_path = os.path.join(training_dir, 'transformed_metadata')
    transformed_metadata = metadata_io.read_metadata(transformed_metadata_path)

    transformed_data_paths = os.path.join(training_dir, 'features_train*')

    pattern = transformed_data_paths[0] if len(
        transformed_data_paths) == 1 else transformed_data_paths
    return input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern=pattern,
        training_batch_size=batch_size,
        label_keys=[TARGET_FEATURE_COLUMN],
        reader=gzip_reader_fn,
        key_feature_name=KEY_FEATURE_COLUMN,
        reader_num_threads=4,
        queue_capacity=batch_size * 2,
        randomize_input=True,
        num_epochs=2)


def eval_input_fn(training_dir, eval_batch_size):
    transformed_metadata_path = os.path.join(training_dir, 'transformed_metadata')
    transformed_metadata = metadata_io.read_metadata(transformed_metadata_path)

    transformed_data_paths = os.path.join(training_dir, 'features_eval*')

    pattern = transformed_data_paths[0] if len(
        transformed_data_paths) == 1 else transformed_data_paths
    return input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern=pattern,
        training_batch_size=eval_batch_size,
        label_keys=[TARGET_FEATURE_COLUMN],
        reader=gzip_reader_fn,
        key_feature_name=KEY_FEATURE_COLUMN,
        reader_num_threads=4,
        queue_capacity=eval_batch_size * 2,
        randomize_input=False,
        num_epochs=None)


if __name__ == '__main__':


    env = sagemaker_containers.training_env()

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--model_type', type=str, default='linear')
    parser.add_argument('--dataset', type=str, default='kaggle')

    parser.add_argument(
        '--ignore_crosses',
        action='store_true',
        default=False,
        help='Whether to ignore crosses (linear model only).')

    parser.add_argument(
        '--hidden_units',
        nargs='*',
        help='List of hidden units per layer. All layers are fully connected. Ex.'
             '`64 32` means first layer has 64 nodes and second one has 32.',
        default=[512],
        type=int)

    parser.add_argument(
        '--batch_size',
        default=30000,
        type=int)

    parser.add_argument(
        '--eval_batch_size',
        default=5000,
        type=int)

    parser.add_argument(
        '--training_steps',
        default=100000,
        type=int)

    parser.add_argument(
        '--eval_steps',
        default=100,
        type=int)

    parser.add_argument(
        '--inter_op_parallelism_threads',
        type=int)

    parser.add_argument('--training_channel', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args = parser.parse_args()

    dat = vars(args)

    config_proto = tf.ConfigProto()

    if os.environ['OMP_NUM_THREADS']:
        config_proto.intra_op_parallelism_threads = int(os.environ['OMP_NUM_THREADS'])
        config_proto.inter_op_parallelism_threads = args.inter_op_parallelism_threads

    dat['config_proto.intra_op_parallelism_threads'] = config_proto.intra_op_parallelism_threads
    dat['config_proto.inter_op_parallelism_threads'] = config_proto.inter_op_parallelism_threads

    dat['tf.VERSION'] = tf.VERSION
    dat['tf.GIT_VERSION'] = tf.GIT_VERSION
    dat['tf.COMPILER_VERSION'] = tf.COMPILER_VERSION
    dat['tf.CXX11_ABI_FLAG'] = tf.CXX11_ABI_FLAG
    dat['tf.MONOLITHIC_BUILD'] = tf.MONOLITHIC_BUILD
    dat['tf.GRAPH_DEF_VERSION'] = tf.GRAPH_DEF_VERSION
    dat['tf.GRAPH_DEF_VERSION_MIN_CONSUMER'] = tf.GRAPH_DEF_VERSION_MIN_CONSUMER
    dat['tf.GRAPH_DEF_VERSION_MIN_PRODUCER'] = tf.GRAPH_DEF_VERSION_MIN_PRODUCER

    for k, v in os.environ.items():
        dat[k] = v
    logger.info(
        '===============================================================================\n'
        '%s'
        '===============================================================================\n'
        % json.dumps(dat, indent=4))

    run_config = tf.estimator.RunConfig(session_config=config_proto)

    estimator = estimator_fn(run_config, args.dataset, not args.ignore_crosses, args.model_type,
                             args.hidden_units)

    train_spec = tf.estimator.TrainSpec(train_input_fn(args.training_channel, args.batch_size),
                                        max_steps=args.training_steps)

    eval_spec = tf.estimator.EvalSpec(eval_input_fn(args.training_channel, args.eval_batch_size),
                                      steps=args.eval_steps)

    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
