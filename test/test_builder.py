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

train_instance_type = 'ml.c5.9xlarge'


@pytest.mark.parametrize('framework_version', ['1.6.0'])
def test_build(framework_version):
    estimator = Estimator(name='tensorflow',
                          framework_version=framework_version,
                          py_version='py2',
                          train_instance_count=1,
                          train_instance_type='ml.c5.9xlarge',
                          source_dir='trainer',
                          entry_point='train.py')

    estimator.build_and_push()
