import logging
import os

import pytest

logging.getLogger('boto3').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('sagemaker.local.local_session').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)
logging.getLogger('sagemaker.local.image').setLevel(logging.DEBUG)

dir_path = os.path.dirname(os.path.realpath(__file__))


def pytest_addoption(parser):
    parser.addoption('--dataset', type=str, default='large', choices=['kaggle', 'large'])
    parser.addoption('--dataset-location', type=str)
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--framework-version', default='1.7.0')
    parser.addoption('--instance-type', default='ml.c5.9xlarge')
    parser.addoption('--sagemaker-role', type=str)
    parser.addoption('--num-parameter-servers', type=int, default=2)
    parser.addoption('--num-instances', type=int, default=3)
    parser.addoption('--checkpoint-path', type=str, default=None)
    parser.addoption('--kmp-blocktime', type=int, default=25)
    parser.addoption('--inter-op-parallelism-threads', type=int, default=10)
    parser.addoption('--s3-use-https', action="store_true")
    parser.addoption('--s3-verify-ssl', action="store_true")
    parser.addoption('--wait', action="store_true")
    parser.addoption('-S', default=True)
    parser.addoption('-V', default=True)


@pytest.fixture(scope='session')
def checkpoint_path(request):
    option = request.config.getoption('--checkpoint-path')
    return option


@pytest.fixture(scope='session')
def dataset(request):
    option = request.config.getoption('--dataset')

    if option is None:
        raise ValueError('Missing argument --dataset')
    return option


@pytest.fixture(scope='session')
def dataset_location(request):
    option = request.config.getoption('--dataset-location')

    if option is None:
        raise ValueError('Missing argument --dataset-location')
    return option


@pytest.fixture(scope='session')
def kmp_blocktime(request):
    option = request.config.getoption('--kmp-blocktime')

    if option is None:
        raise ValueError('Missing argument --kmp-blocktime')
    return option


@pytest.fixture(scope='session')
def s3_use_https(request):
    option = request.config.getoption('--s3-use-https')

    if option is None:
        raise ValueError('Missing argument --s3-use-https')
    return option


@pytest.fixture(scope='session')
def wait(request):
    option = request.config.getoption('--wait')

    if option is None:
        raise ValueError('Missing argument --wait')
    return option


@pytest.fixture(scope='session')
def s3_verify_ssl(request):
    option = request.config.getoption('--s3-verify-ssl')

    if option is None:
        raise ValueError('Missing argument --s3-verify-ssl')
    return option


@pytest.fixture(scope='session')
def inter_op_parallelism_threads(request):
    option = request.config.getoption('--inter-op-parallelism-threads')

    if option is None:
        raise ValueError('Missing argument --inter-op-parallelism-threads')
    return option


@pytest.fixture(scope='session')
def num_parameter_servers(request):
    option = request.config.getoption('--num-parameter-servers')

    if option is None:
        raise ValueError('Missing argument --num-parameter-servers')
    return option


@pytest.fixture(scope='session')
def placeholder_bucket(region, instance_type):
    if instance_type == 'local':
        return 'file://%s' % dir_path
    else:
        return 's3://sagemaker-sample-data-%s/spark/mnist/train' % region


@pytest.fixture(scope='session')
def region(request):
    option = request.config.getoption('--region')

    if option is None:
        raise ValueError('Missing argument --region')
    return option


@pytest.fixture(scope='session')
def framework_version(request):
    option = request.config.getoption('--framework-version')

    if option is None:
        raise ValueError('Missing argument --framework-version')
    return option


@pytest.fixture(scope='session')
def instance_type(request):
    option = request.config.getoption('--instance-type')

    if option is None:
        raise ValueError('Missing argument --instance-type')
    return option


@pytest.fixture(scope='session')
def num_instances(request):
    option = request.config.getoption('--num-instances')

    if option is None:
        raise ValueError('Missing argument --num-instances')
    return option


@pytest.fixture(scope='session')
def sagemaker_role(request):
    option = request.config.getoption('--sagemaker-role')

    if option is None:
        raise ValueError('Missing argument --sagemaker-role')
    return option
