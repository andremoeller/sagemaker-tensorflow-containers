from __future__ import absolute_import

import logging

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


@pytest.mark.parametrize('framework_version',
                         ['1.6.0', '1.6.0.vanilla', '1.8.0', '1.8.0.vanilla'])
def test_build(framework_version):
    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          train_instance_type='ml.c5.9xlarge',
                          source_dir='trainer',
                          build_image=True,
                          entry_point='train.py')

    input_dir = 'file:///tmp/data/data'

    estimator.fit(input_dir)


@pytest.mark.parametrize('use_https', [
    # '1',
    '0'])
@pytest.mark.parametrize('s3_verify_ssl',
                         # ['1', '0'])
                         # ['1'])
                         ['0'])
@pytest.mark.parametrize('omp_proc_bind',
                         ['true', 'false'])
# ['true'])
# @pytest.mark.parametrize('omp_proc_bind', ['true'])
@pytest.mark.parametrize('kmp_blocktime', ['0', '1', '10', '100', '200'])
@pytest.mark.parametrize('kmp_affinity', [
    # 'noverbose,warnings,respect,granularity=core,duplicates,none',
    # 'noverbose,warnings,respect,granularity=fine,duplicates,scatter,0,0',
    'granularity=fine,compact,1,0',
    'granularity=fine,verbose,compact,1,0'
])
@pytest.mark.parametrize('batch_size', [
    30000,
    # 15000,
    # 10000,
    # 5000,
    # 4000,
    # 6000,
    # 300000,
    # 3000,
    # 300
])
@pytest.mark.parametrize('omp_num_threads',
                         # ['36', None]
                         ['36']
                         )
@pytest.mark.parametrize('inter_op_parallelism_threads',
                         # ['36', None]
                         [1, 2, 4, 6, 8, 10]
                         )
@pytest.mark.parametrize('framework_version',
                         # ['1.6.0', '1.8.0'] ## '1.6.0.vanilla', '1.8.0' #, '1.8.0.vanilla' ])
                         ['1.6.0']  ## '1.6.0.vanilla', '1.8.0' #, '1.8.0.vanilla' ])
                         )
def test_tuning(use_https, s3_verify_ssl, omp_proc_bind, kmp_blocktime,
                kmp_affinity, batch_size, framework_version, omp_num_threads,
                inter_op_parallelism_threads):
    hps = {
        'batch_size':                   batch_size,
        'inter_op_parallelism_threads': inter_op_parallelism_threads,
        'sagemaker_env_vars':           {
            'S3_REQUEST_TIMEOUT_MSEC': '60000',
            'S3_REGION':               'us-west-2',
            # 'S3_USE_HTTPS':            use_https,
            'S3_USE_HTTPS':            s3_verify_ssl,
            'S3_VERIFY_SSL':           s3_verify_ssl,
            'OMP_PROC_BIND':           omp_proc_bind,
            'KMP_BLOCKTIME':           kmp_blocktime,
            'KMP_AFFINITY':            kmp_affinity,
            'KMP_SETTINGS':            '1'
        }
    }

    if omp_num_threads:
        hps['sagemaker_env_vars']['OMP_NUM_THREADS'] = omp_num_threads
    # env_vars['TF_CPP_MIN_VLOG_LEVEL'] = '3'# maximum log
    # https://stackoverflow.com/questions/45141895/how-does-tensorflows-vlog-work env_vars[
    # 'TF_CPP_MIN_LOG_LEVEL'] = '0' # maximum log
    # env_vars["KMP_SETTINGS"] = '1' # print vars

    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='train.py')

    if train_instance_type == 'local':
        input_dir = 'file:///tmp/data/data'
    else:
        input_dir = "s3://sagemaker-us-west-2-369233609183/datasets/criteo-dataset"

    estimator.fit(input_dir, wait=False)


@pytest.mark.parametrize('s3_verify_ssl', ['0'])
@pytest.mark.parametrize('omp_proc_bind', ['true'])
@pytest.mark.parametrize('kmp_blocktime', ['1'])
@pytest.mark.parametrize('kmp_affinity', ['granularity=fine,verbose,compact,1,0'])
@pytest.mark.parametrize('batch_size', [30000])
@pytest.mark.parametrize('omp_num_threads', ['36'])
@pytest.mark.parametrize('inter_op_parallelism_threads', [2])
@pytest.mark.parametrize('framework_version', ['1.6.0'])
def test_snap_db(s3_verify_ssl, omp_proc_bind, kmp_blocktime,
                 kmp_affinity, batch_size, framework_version, omp_num_threads,
                 inter_op_parallelism_threads):
    hps = {
        'batch_size':                   batch_size,
        'inter_op_parallelism_threads': inter_op_parallelism_threads,
        'sagemaker_env_vars':           {
            'S3_REQUEST_TIMEOUT_MSEC': '60000',
            'S3_REGION':               'us-west-2',
            'S3_USE_HTTPS':            s3_verify_ssl,
            'S3_VERIFY_SSL':           s3_verify_ssl,
            'OMP_PROC_BIND':           omp_proc_bind,
            'KMP_BLOCKTIME':           kmp_blocktime,
            'KMP_AFFINITY':            kmp_affinity,
        }
    }

    if omp_num_threads:
        hps['sagemaker_env_vars']['OMP_NUM_THREADS'] = omp_num_threads

    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          hyperparameters=hps,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          train_volume_size=100,
                          entry_point='train.py')

    input_dir = 's3://sagemaker-snap-data/criteo_20180524_182711'

    estimator.fit(input_dir)
