from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

IRIS_TRAINING = os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'iris_training.csv')
IRIS_TEST = os.path.join(os.environ['SM_CHANNEL_TEST'], 'iris_test.csv')

FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


def input_fn(file_name, num_data, batch_size, is_training):
    """Creates an input_fn required by Estimator train/evaluate."""

    def _parse_csv(rows_string_tensor):
        """Takes the string input tensor and returns tuple of (features, labels)."""
        # Last dim is the label.
        num_features = len(FEATURE_KEYS)
        num_columns = num_features + 1
        columns = tf.decode_csv(rows_string_tensor,
                                record_defaults=[[]] * num_columns)
        features = dict(zip(FEATURE_KEYS, columns[:num_features]))
        labels = tf.cast(columns[num_features], tf.int32)
        return features, labels

    def _input_fn():
        dataset = tf.data.TextLineDataset([file_name])
        # Skip the first line (which does not have data).
        dataset = dataset.skip(1)
        dataset = dataset.map(_parse_csv)

        if is_training:
            dataset = dataset.shuffle(num_data)
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return _input_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    feature_columns = [tf.feature_column.numeric_column(key, shape=1) for key in FEATURE_KEYS]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10], n_classes=3)

    train_input_fn = input_fn(IRIS_TRAINING, num_data=32, batch_size=32, is_training=True)

    classifier.train(input_fn=train_input_fn, steps=400)

    test_input_fn = input_fn(IRIS_TEST, num_data=32, batch_size=32, is_training=False)

    scores = classifier.evaluate(input_fn=test_input_fn)

    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
    tf.app.run()
