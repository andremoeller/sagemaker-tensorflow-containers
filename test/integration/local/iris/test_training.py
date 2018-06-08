# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import logging
import os

from test.estimator import Estimator

logger = logging.basicConfig()

dir_path = os.path.dirname(os.path.realpath(__file__))
py_version = 'py2'
train_instance_type = 'local'


def test_training():
    estimator = Estimator(name='tensorflow',
                          framework_version='1.6.0',
                          py_version=py_version,
                          train_instance_count=1,
                          train_instance_type=train_instance_type,
                          source_dir=dir_path, build_image=True,
                          entry_point='iris.py')

    estimator.fit({})

    assert os.path.exists(os.path.join(estimator.output_dir, 'success'))
    assert os.path.exists(os.path.join(estimator.model_data, 'checkpoint'))
    assert os.path.exists(os.path.join(estimator.model_data, 'graph.pbtxt'))
    assert os.path.exists(os.path.join(estimator.model_data, 'eval'))
