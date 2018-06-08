from __future__ import absolute_import

import logging

from benchmarks.criteo import DATASETS_CRITEO_SMALLCLICKS
from test.estimator import Estimator

logging.getLogger('boto3').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('sagemaker.local.local_session').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)
logging.getLogger('sagemaker.local.image').setLevel(logging.DEBUG)

role = 'SageMakerRole'

train_instance_type = 'local'
train_instance_type = 'ml.c5.9xlarge'


def test_chan():
    estimator = Estimator(name='tensorflow',
                          framework_version='1.6.0',
                          py_version='py2',
                          train_instance_count=1,
                          train_instance_type=train_instance_type,
                          source_dir='trainer',
                          build_image=False,
                          entry_point='task.py')

    input_dir = DATASETS_CRITEO_SMALLCLICKS

    estimator.fit(input_dir, wait=False)
