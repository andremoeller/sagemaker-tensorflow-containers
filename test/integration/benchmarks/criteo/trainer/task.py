from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import math
import os
import sys

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.INFO)

logger = logging.getLogger('user-script')  # initialize logging class
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(formatter)
logger.addHandler(sh)

logging.getLogger('tensorflow').addHandler(sh)

import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn import learn_runner

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
        NUM_EXAMPLES: 45 * 1e6,
        L2_REGULARIZATION: 60,
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
        NUM_EXAMPLES: 4 * 1e9,
        L2_REGULARIZATION: 500,
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


def get_vocab_sizes():
    """Read vocabulary sizes from the metadata."""
    return {FORMAT_CATEGORICAL_FEATURE_ID.format(index): int(10 * 1000) for index in range(14, 40)}


def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def create_parser():

    if os.environ.get('S3_CHANNEL', None):
        training_dir = os.environ['S3_CHANNEL']
    else:
        training_dir = os.environ['SM_CHANNEL_TRAINING']

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
        '--train_data_paths', type=str, action='append',
        default=os.path.join(training_dir, 'features_train*'))

    parser.add_argument(
        '--eval_data_paths', type=str, action='append',
        default=os.path.join(training_dir, 'features_eval*'))

    parser.add_argument(
        '--batch_size',
        default=30000,
        type=int)

    parser.add_argument(
        '--eval_batch_size',
        default=5000,
        type=int)

    parser.add_argument(
        '--train_steps',
        default=0,
        type=int)

    parser.add_argument(
        '--eval_steps',
        default=os.environ.get('SM_HP_EVAL_STEPS', 100),
        type=int)

    parser.add_argument('--transformed_metadata_path', type=str,
                        default=os.path.join(training_dir, 'transformed_metadata'))

    parser.add_argument('--l2_regularization', help='L2 Regularization', type=int)

    parser.add_argument('--num_epochs', help='Number of epochs', default=5, type=int)

    parser.add_argument('--inter_op_parallelism_threads', type=int)

    parser.add_argument('--training_channel', type=str, default=training_dir)

    return parser


def feature_columns(config, model_type, vocab_sizes, use_crosses):
    """Return the feature columns with their names and types."""
    result = []
    boundaries = [1.5 ** j - 0.51 for j in range(40)]

    # TODO(b/35300113): Reduce the range and other duplication between this and
    # preprocessing.

    # TODO(b/35300113): Can iterate over metadata so that we don't need to
    # re-define the schema here?

    for index in range(1, 14):
        column = tf.contrib.layers.bucketized_column(
            tf.contrib.layers.real_valued_column(
                FORMAT_INT_FEATURE.format(index),
                dtype=tf.int64),
            boundaries)
        result.append(column)

    if model_type == LINEAR:
        for index in range(14, 40):
            column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
            vocab_size = vocab_sizes[column_name]
            column = tf.contrib.layers.sparse_column_with_integerized_feature(
                column_name, vocab_size, combiner='sum')
            result.append(column)
        if use_crosses:
            for cross in config[CROSSES]:
                column = tf.contrib.layers.crossed_column(
                    [result[index - 1] for index in cross],
                    hash_bucket_size=CROSS_HASH_BUCKET_SIZE,
                    hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
                    combiner='sum')
                result.append(column)
    elif model_type == DEEP:
        for index in range(14, 40):
            column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
            vocab_size = vocab_sizes[column_name]
            column = tf.contrib.layers.sparse_column_with_integerized_feature(
                column_name, vocab_size, combiner='sum')
            embedding_size = int(math.floor(6 * vocab_size ** 0.25))
            embedding = tf.contrib.layers.embedding_column(column,
                                                           embedding_size,
                                                           combiner='mean')
            result.append(embedding)

    return result


def get_experiment_fn(train_steps, eval_steps, num_epochs, ignore_crosses, dataset, model_type,
                      l2_regularization, hidden_units,
                      transformed_metadata_path,
                      train_data_paths, batch_size, eval_data_paths):
    """Wrap the get experiment function to provide the runtime arguments."""
    vocab_sizes = get_vocab_sizes()
    use_crosses = not ignore_crosses
    config = PIPELINE_CONFIG.get(dataset)

    l2_regularization = l2_regularization or config[L2_REGULARIZATION]
    train_set_size = config[NUM_EXAMPLES]

    def get_experiment(run_config, hparams):

        columns = feature_columns(config, model_type, vocab_sizes, use_crosses)

        cluster = run_config.cluster_spec
        num_table_shards = max(1, run_config.num_ps_replicas * 3)
        num_partitions = max(1, 1 + cluster.num_tasks('worker') \
            if cluster and 'worker' in cluster.jobs else 0)

        if model_type == LINEAR:
            estimator = tf.contrib.learn.LinearClassifier(
                config=run_config,
                feature_columns=columns,
                optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
                    example_id_column=KEY_FEATURE_COLUMN,
                    symmetric_l2_regularization=l2_regularization,
                    num_loss_partitions=num_partitions,  # workers
                    num_table_shards=num_table_shards))  # ps
        else:
            estimator = tf.contrib.learn.DNNClassifier(
                hidden_units=hidden_units,
                feature_columns=columns,
                config=run_config)

        transformed_metadata = metadata_io.read_metadata(transformed_metadata_path)

        _train_input_fn = get_transformed_reader_input_fn(
            transformed_metadata, train_data_paths, batch_size,
            tf.contrib.learn.ModeKeys.TRAIN)

        _eval_input_fn = get_transformed_reader_input_fn(
            transformed_metadata, eval_data_paths, batch_size,
            tf.contrib.learn.ModeKeys.EVAL)

        return tf.contrib.learn.Experiment(
            estimator=estimator,
            train_steps=(train_steps or
                         num_epochs * train_set_size // batch_size),
            eval_steps=eval_steps,
            train_input_fn=_train_input_fn,
            eval_input_fn=_eval_input_fn,
            min_eval_frequency=500)

    # Return a function to create an Experiment.
    return get_experiment


def get_transformed_reader_input_fn(transformed_metadata,
                                    transformed_data_paths,
                                    batch_size,
                                    mode):
    """Wrap the get input features function to provide the runtime arguments."""
    # https://github.com/tensorflow/transform/blob/master/tensorflow_transform/saved/input_fn_maker.py#L548
    return input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern=(
            transformed_data_paths[0] if len(transformed_data_paths) == 1
            else transformed_data_paths),
        training_batch_size=batch_size,
        label_keys=[TARGET_FEATURE_COLUMN],
        reader=gzip_reader_fn,
        key_feature_name=KEY_FEATURE_COLUMN,
        reader_num_threads=4,
        queue_capacity=batch_size * 2,
        randomize_input=(mode != tf.contrib.learn.ModeKeys.EVAL),
        num_epochs=(1 if mode == tf.contrib.learn.ModeKeys.EVAL else None))


def main():
    args = create_parser().parse_args()

    info = vars(args)

    config_proto = tf.ConfigProto()

    if os.environ.get('OMP_NUM_THREADS', None):
        config_proto.intra_op_parallelism_threads = int(os.environ['OMP_NUM_THREADS'])
        config_proto.inter_op_parallelism_threads = args.inter_op_parallelism_threads

    info['config_proto.intra_op_parallelism_threads'] = config_proto.intra_op_parallelism_threads
    info['config_proto.inter_op_parallelism_threads'] = config_proto.inter_op_parallelism_threads

    info['tf.VERSION'] = tf.VERSION
    info['tf.GIT_VERSION'] = tf.GIT_VERSION
    info['tf.COMPILER_VERSION'] = tf.COMPILER_VERSION
    info['tf.CXX11_ABI_FLAG'] = tf.CXX11_ABI_FLAG
    info['tf.MONOLITHIC_BUILD'] = tf.MONOLITHIC_BUILD
    info['tf.GRAPH_DEF_VERSION'] = tf.GRAPH_DEF_VERSION
    info['tf.GRAPH_DEF_VERSION_MIN_CONSUMER'] = tf.GRAPH_DEF_VERSION_MIN_CONSUMER
    info['tf.GRAPH_DEF_VERSION_MIN_PRODUCER'] = tf.GRAPH_DEF_VERSION_MIN_PRODUCER

    for k, v in os.environ.items():
        info[k] = v
    logger.info(
        '==============================   TRAINING INFO   =============================\n\n'
        '%s\n\n'
        '===============================================================================\n'
        % json.dumps(info, indent=4))

    experiment_fn = get_experiment_fn(args.train_steps, args.eval_steps, args.num_epochs,
                                      args.ignore_crosses, args.dataset, args.model_type,
                                      args.l2_regularization, args.hidden_units,
                                      args.transformed_metadata_path,
                                      args.train_data_paths, args.batch_size, args.eval_data_paths)

    run_config = tf.contrib.learn.RunConfig(session_config=config_proto)

    learn_runner.run(experiment_fn=experiment_fn,
                     run_config=run_config)

if __name__ == '__main__':
    main()
