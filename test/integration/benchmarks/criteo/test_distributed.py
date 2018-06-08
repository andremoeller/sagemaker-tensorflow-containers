from __future__ import absolute_import

import logging
import os

from benchmarks.criteo import (DATASETS_CRITEO_SMALLCLICKS, BOGUS_S3_FILE,
                               DATASET_CRITEO_LARGECLICKS, LOCAL_DATA)
from test.estimator import Estimator
import pytest

from sagemaker import Session

logging.getLogger('boto3').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('sagemaker.local.local_session').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)
logging.getLogger('sagemaker.local.image').setLevel(logging.DEBUG)

role = 'SageMakerRole'

# train_instance_type = 'local'
train_instance_type = 'ml.c5.9xlarge'

default_bucket = Session().default_bucket


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
@pytest.mark.parametrize('num_instances', [1, 2, 3, 4, 5, 10, 20])
@pytest.mark.parametrize('framework_version', ['1.8.0'])
def test_tuning_final(use_https, s3_verify_ssl, omp_proc_bind, kmp_blocktime,
                      kmp_affinity, batch_size, framework_version, omp_num_threads,
                      inter_op_parallelism_threads, num_instances):
    job_name = ('%s-kmp_blocktime%s-inter-op-threads-%s' % (
        framework_version,
        kmp_blocktime, inter_op_parallelism_threads)).replace('.', '-').replace('_', '-')

    hps = {
        'batch_size':                   batch_size,
        'sagemaker_model_dir':          os.path.join(
            default_bucket,
            'benchmarks',
            'distributed-vpc',
            '%s_instances' % num_instances,
        ),
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
                          train_instance_count=num_instances,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    if train_instance_type == 'local':
        input_dir = 'file:///tmp/data/preproc'
    else:
        input_dir = DATASETS_CRITEO_SMALLCLICKS

    estimator.fit(input_dir, wait=False)


@pytest.mark.parametrize('kmp_blocktime', ['25'])
@pytest.mark.parametrize('batch_size', [30000])
@pytest.mark.parametrize('inter_op_parallelism_threads', [10])
@pytest.mark.parametrize('num_instances', [30])
@pytest.mark.parametrize('num_ps', [30, 17, 10])
@pytest.mark.parametrize('framework_version', ['1.7.0'])
@pytest.mark.parametrize('read_from_s3', [True])
def test_ps_server(kmp_blocktime, batch_size, framework_version, read_from_s3,
                   inter_op_parallelism_threads, num_instances, num_ps):
    hps = {
        'batch_size':                      batch_size,
        'eval_steps':                      1000,
        'sagemaker_num_parameter_servers': min(num_ps, num_instances),
        'sagemaker_model_dir':             os.path.join(
            default_bucket,
            'benchmark-16TB',
            '%s-instances-%snum_ps-%sversion-s3%s' % (num_instances, num_ps, framework_version,
                                                      read_from_s3)
        ),
        'dataset':                         'large',
        'model_type':                      'linear',
        'l2_regularization':               3000,
        'inter_op_parallelism_threads':    inter_op_parallelism_threads,
        'sagemaker_env_vars':              {

            'KMP_BLOCKTIME': kmp_blocktime,

        }
    }

    if read_from_s3:
        input_dir = BOGUS_S3_FILE

        hps['sagemaker_env_vars']['S3_CHANNEL'] = DATASET_CRITEO_LARGECLICKS

    elif train_instance_type == 'local':
        input_dir = LOCAL_DATA
    else:
        input_dir = DATASETS_CRITEO_SMALLCLICKS

    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=num_instances,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    estimator.fit(input_dir, wait=False)
