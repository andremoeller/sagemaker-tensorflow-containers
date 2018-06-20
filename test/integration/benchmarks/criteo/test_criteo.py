from __future__ import absolute_import

import os

from sagemaker import Session
from sagemaker.tensorflow import TensorFlow

from sagemaker.session import s3_input

default_bucket = Session().default_bucket

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_benchmark(framework_version, instance_type, placeholder_bucket, dataset,
                   num_parameter_servers, inter_op_parallelism_threads, s3_verify_ssl,
                   s3_use_https, kmp_blocktime, sagemaker_role, dataset_location, region,
                   wait, num_instances, checkpoint_path):

    hyperparameters = {
        # sets the number of parameter servers in the cluster.
        'sagemaker_num_parameter_servers': num_parameter_servers,
        's3_channel':                      dataset_location,
        'batch_size':                      30000,
        #'train_steps': 10,
        #'eval_steps': 10,
        #'num_epochs': 1,
        'dataset':                         dataset,
        'model_type':                      'linear',
        'l2_regularization':               100,

        # see https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu
        # Best value for this model is 10, default value in the container is 0.
        # 0 sets the value to the number of logical cores.
        'inter_op_parallelism_threads':    inter_op_parallelism_threads,

        # environment variables that will be written to the container before training starts
        'sagemaker_env_vars':              {
            # True uses HTTPS, uses HTTP otherwise. Default false
            # see https://docs.aws.amazon.com/sdk-for-cpp/v1/developer-guide/client-config.html
            'S3_USE_HTTPS':  s3_verify_ssl,
            # True verifies SSL. Default false
            'S3_VERIFY_SSL': s3_use_https,
            # Sets the time, in milliseconds, that a thread should wait, after completing the
            # execution of a parallel region, before sleeping. Default 0
            # see https://github.com/tensorflow/tensorflow/blob/faff6f2a60a01dba57cf3a3ab832279dbe174798/tensorflow/docs_src/performance/performance_guide.md#tuning-mkl-for-the-best-performance
            'KMP_BLOCKTIME': kmp_blocktime
        }
    }

    tf = TensorFlow(entry_point='task.py',
                    source_dir=os.path.join(dir_path, 'trainer'),
                    train_instance_count=num_instances,
                    train_instance_type=instance_type,
                    # pass in your own SageMaker role
                    checkpoint_path=checkpoint_path,
                    role=sagemaker_role,
                    hyperparameters=hyperparameters)

    # This points to the prototype images.
    #tf.train_image = lambda: '520713654638.dkr.ecr.%s.amazonaws.com/sagemaker-tensorflow:' \
    #                         '%s-cpu-py2-script-mode-preview' % (region, framework_version)

    tf.train_image = lambda: '038453126632.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.7.0-cpu-py2'

    #gzip_input = s3_input(placeholder_bucket, compression='Gzip')

    tf.fit({'training': placeholder_bucket}, wait=wait)
