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
import shutil
import subprocess

from sagemaker.estimator import Framework
from sagemaker.fw_utils import create_image_uri

logger = logging.getLogger(__name__)


class Estimator(Framework):
    def __init__(self, name, framework_version, entry_point, py_version,
                 source_dir=None, hyperparameters=None, build_image=True, **kwargs):

        super(Estimator, self).__init__(
            entry_point, source_dir, hyperparameters,
            role='SageMakerRole', container_log_level=logging.INFO, **kwargs)

        self.image = None
        self.py_version = py_version
        self.framework_version = framework_version
        self.name = name
        self.build_image = build_image

    @classmethod
    def _from_training_job(cls, init_params, hyperparameters, image, sagemaker_session):
        pass

    def create_model(self, **kwargs):
        pass

    def train_image(self):
        return self.build_and_push()

    def build_and_push(self):
        identity = self.sagemaker_session.boto_session.client('sts').get_caller_identity()
        tag = create_image_uri(self.sagemaker_session.boto_session.region_name, self.name,
                               self.train_instance_type, self.framework_version, self.py_version,
                               identity['Account'])
        was_build = False
        if self.build_image:
            self._build(tag)
            self.build_image = False
            was_build = True
        if was_build and self.train_instance_type.startswith('ml.'):
            self._push(tag)
        return tag

    @property
    def output_dir(self):
        return os.path.join(self.model_data, '..', 'output')

    def _build(self, tag):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.abspath(os.path.join(dir_path, '..', '..'))

        subprocess.check_call(['python', 'setup.py', 'sdist'], cwd=root_path)

        device_type = 'gpu' if self.train_instance_type[:4] in ['ml.g', 'ml.p'] else 'cpu'

        dockerfile_name = 'Dockerfile.{}'.format(device_type)
        dockerfile_location = os.path.join(root_path, 'docker', self.framework_version,
                                           'final', self.py_version)

        framework_support_installable = os.path.join(root_path, 'dist',
                                                     'sagemaker_tensorflow_container-2.0.0.tar.gz')

        shutil.copy2(framework_support_installable, dockerfile_location)

        dockerfile = os.path.join(dockerfile_location, dockerfile_name)
        cmd = 'docker build -t {} -f {} .'.format(tag, dockerfile).split(' ')
        print(cmd)

        subprocess.check_call(cmd, cwd=dockerfile_location)

        print('created image {}'.format(tag))

    @staticmethod
    def _push(tag):
        cmd = 'aws ecr get-login --no-include-email --region us-west-2'.split(' ')
        login = subprocess.check_output(cmd).strip()

        subprocess.check_call(login.split(' '.encode()))

        subprocess.check_call(cmd)
        cmd = 'docker push {}'.format(tag).split(' ')
        subprocess.check_call(cmd)
