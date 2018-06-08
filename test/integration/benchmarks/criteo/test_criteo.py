from __future__ import absolute_import

import logging
import os
import time

from sagemaker import Session

from benchmarks.criteo import LOCAL_DATA, DATASETS_CRITEO_SMALLCLICKS, BOGUS_S3_FILE
from test.estimator import Estimator
import pytest

logging.getLogger('boto3').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('sagemaker.local.local_session').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)
logging.getLogger('sagemaker.local.image').setLevel(logging.DEBUG)

role = 'SageMakerRole'

# train_instance_type = 'local'
train_instance_type = 'ml.c5.9xlarge'

default_bucket = Session().default_bucket


@pytest.mark.parametrize('framework_version', ['1.7.0'])
def test_build(framework_version):
    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          train_instance_type='ml.c5.9xlarge',
                          source_dir='trainer',
                          entry_point='train.py')

    input_dir = LOCAL_DATA

    estimator.fit(input_dir)


@pytest.mark.parametrize('batch_size', [30000])
@pytest.mark.parametrize('framework_version', ['1.6.0'])
def test_tuning_no_mkl(batch_size, framework_version):
    hps = {
        'batch_size':         batch_size,

        'dataset':            'kaggle',
        'model_type':         'linear',
        'l2_regularization':  100,

        'sagemaker_env_vars': {
            'S3_REQUEST_TIMEOUT_MSEC': '60000',
            'S3_REGION':               'us-west-2',
        }
    }

    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    if train_instance_type == 'local':
        input_dir = LOCAL_DATA
    else:
        input_dir = DATASETS_CRITEO_SMALLCLICKS

    estimator.fit(input_dir, wait=False)


@pytest.mark.parametrize('batch_size', [30000])
@pytest.mark.parametrize('framework_version', ['1.6.0'])
@pytest.mark.parametrize('s3_use_https', ['0'])
@pytest.mark.parametrize('s3_verify_ssl', ['0'])
def test_read_from_s3(batch_size, framework_version, s3_use_https, s3_verify_ssl):
    hps = {
        'batch_size':         batch_size,

        'dataset':            'kaggle',
        'model_type':         'linear',
        'l2_regularization':  100,
        'sagemaker_env_vars': {
            'S3_REQUEST_TIMEOUT_MSEC': '60000',
            'S3_USE_HTTPS':            s3_use_https,
            'S3_VERIFY_SSL':           s3_verify_ssl,
            'S3_REGION':               'us-west-2',
            'S3_CHANNEL':              DATASETS_CRITEO_SMALLCLICKS
        }
    }

    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    fake_channel = BOGUS_S3_FILE

    estimator.fit(fake_channel)


@pytest.mark.parametrize('use_https', ['0'])
@pytest.mark.parametrize('s3_verify_ssl', ['0'])
@pytest.mark.parametrize('omp_proc_bind', ['true'])
@pytest.mark.parametrize('kmp_blocktime', ['1', '4', '10', '25', '50', '100', '200'])
@pytest.mark.parametrize('kmp_affinity', [
    'granularity=fine,verbose,compact,1,0'
])
@pytest.mark.parametrize('batch_size', [30000])
@pytest.mark.parametrize('omp_num_threads', ['36'])
@pytest.mark.parametrize('inter_op_parallelism_threads', [1, 5, 10, 20, 40])
@pytest.mark.parametrize('framework_version', ['1.6.0'])
def test_tuning(use_https, s3_verify_ssl, omp_proc_bind, kmp_blocktime,
                kmp_affinity, batch_size, framework_version, omp_num_threads,
                inter_op_parallelism_threads):
    hps = {
        'batch_size':                   batch_size,

        'dataset':                      'kaggle',
        'model_type':                   'linear',
        'l2_regularization':            100,

        'inter_op_parallelism_threads': inter_op_parallelism_threads,
        'sagemaker_env_vars':           {
            'S3_REQUEST_TIMEOUT_MSEC': '60000',
            'S3_REGION':               'us-west-2',
            'S3_USE_HTTPS':            use_https,
            'S3_VERIFY_SSL':           s3_verify_ssl,
            'OMP_PROC_BIND':           omp_proc_bind,
            'KMP_BLOCKTIME':           kmp_blocktime,
            'KMP_AFFINITY':            kmp_affinity,
            'KMP_SETTINGS':            '1',
            'OMP_NUM_THREADS':         omp_num_threads
        }
    }

    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    if train_instance_type == 'local':
        input_dir = LOCAL_DATA
    else:
        input_dir = DATASETS_CRITEO_SMALLCLICKS

    estimator.fit(input_dir, wait=False)


@pytest.mark.parametrize('use_https', ['0'])
@pytest.mark.parametrize('s3_verify_ssl', ['0'])
@pytest.mark.parametrize('omp_proc_bind', ['true'])
# @pytest.mark.parametrize('kmp_blocktime', ['0', '1', '4', '10', '25', '50', '100', '200',
#                                            '400', 'infinite'])
@pytest.mark.parametrize('kmp_blocktime', ['999'])
@pytest.mark.parametrize('kmp_affinity', [
    'granularity=fine,verbose,compact,1,0'
])
@pytest.mark.parametrize('batch_size', [30000])
@pytest.mark.parametrize('omp_num_threads', ['36'])
@pytest.mark.parametrize('inter_op_parallelism_threads', [10])
@pytest.mark.parametrize('framework_version', ['1.6.0', '1.6.0.vanilla', '1.8.0', '1.8.0.vanilla'])
def test_tuning_final(use_https, s3_verify_ssl, omp_proc_bind, kmp_blocktime,
                      kmp_affinity, batch_size, framework_version, omp_num_threads,
                      inter_op_parallelism_threads):
    job_name = ('%s-kmp_blocktime%s-inter-op-threads-%s' % (
        framework_version,
        kmp_blocktime, inter_op_parallelism_threads)).replace('.', '-').replace('_', '-')

    hps = {
        'batch_size':                   batch_size,
        'model_dir':                    os.path.join(
            default_bucket,
            'mkl-benchmarks',
            '%s-%s' % (job_name, time.time()),
            'checkpoints'),
        'dataset':                      'kaggle',
        'model_type':                   'linear',
        'l2_regularization':            100,

        'inter_op_parallelism_threads': inter_op_parallelism_threads,
        'sagemaker_env_vars':           {
            'S3_REQUEST_TIMEOUT_MSEC': '60000',
            'S3_REGION':               'us-west-2',
            'S3_USE_HTTPS':            use_https,
            'S3_VERIFY_SSL':           s3_verify_ssl,
            'OMP_PROC_BIND':           omp_proc_bind,
            'KMP_BLOCKTIME':           kmp_blocktime,
            'KMP_AFFINITY':            kmp_affinity,
            'KMP_SETTINGS':            '1',
            'OMP_NUM_THREADS':         omp_num_threads
        }
    }

    estimator = Estimator(name='tensorflow',
                          base_job_name=job_name,
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    if train_instance_type == 'local':
        input_dir = LOCAL_DATA
    else:
        input_dir = DATASETS_CRITEO_SMALLCLICKS

    estimator.fit(input_dir, wait=False)


@pytest.mark.parametrize('use_https', ['0'])
@pytest.mark.parametrize('s3_verify_ssl', ['0'])
@pytest.mark.parametrize('omp_proc_bind', ['true'])
@pytest.mark.parametrize('kmp_blocktime', ['25'])
@pytest.mark.parametrize('kmp_affinity', [
    'granularity=fine,verbose,compact,1,0'
])
@pytest.mark.parametrize('batch_size', [30000])
@pytest.mark.parametrize('omp_num_threads', ['36'])
@pytest.mark.parametrize('inter_op_parallelism_threads', [10])
@pytest.mark.parametrize('framework_version', ['1.6.0', '1.6.0.vanilla', '1.8.0', '1.8.0.vanilla'])
def test_tuning_final(use_https, s3_verify_ssl, omp_proc_bind, kmp_blocktime,
                      kmp_affinity, batch_size, framework_version, omp_num_threads,
                      inter_op_parallelism_threads):
    job_name = ('%s-kmp_blocktime%s-inter-op-threads-%s' % (
        framework_version,
        kmp_blocktime, inter_op_parallelism_threads)).replace('.', '-').replace('_', '-')

    hps = {
        'batch_size':                   batch_size,
        'model_dir':                    os.path.join(
            default_bucket,
            'mkl-benchmarks',
            '%s-%s' % (job_name, time.time()),
            'checkpoints'),
        'dataset':                      'kaggle',
        'model_type':                   'linear',
        'l2_regularization':            100,

        'inter_op_parallelism_threads': inter_op_parallelism_threads,
        'sagemaker_env_vars':           {
            'S3_REQUEST_TIMEOUT_MSEC': '60000',
            'S3_REGION':               'us-west-2',
            'S3_USE_HTTPS':            use_https,
            'S3_VERIFY_SSL':           s3_verify_ssl,
            'OMP_PROC_BIND':           omp_proc_bind,
            'KMP_BLOCKTIME':           kmp_blocktime,
            'KMP_AFFINITY':            kmp_affinity,
            'KMP_SETTINGS':            '1',
            'OMP_NUM_THREADS':         omp_num_threads
        }
    }

    estimator = Estimator(name='tensorflow',
                          base_job_name=job_name,
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    if train_instance_type == 'local':
        input_dir = LOCAL_DATA
    else:
        input_dir = DATASETS_CRITEO_SMALLCLICKS

    estimator.fit(input_dir, wait=False)
