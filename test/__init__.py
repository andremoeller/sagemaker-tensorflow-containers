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

import collections
import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile

os.environ['base_dir'] = os.path.join(tempfile.mkdtemp(), 'opt', 'ml')

from typing import List, Dict, Optional
from sagemaker_containers.beta.framework import env, modules, params

logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))

USER_SCRIPTS_PATH = os.path.join(dir_path, 'user_scripts')
DATASETS_PATH = os.path.join(dir_path, 'datasets')


def uninstall(module_name):
    try:
        cmd = [modules.python_executable(), '-m', 'pip', 'uninstall', '-y', module_name]

        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        pass


@contextlib.contextmanager
def prepare_training(module_name, channels=None, hosts=None):
    #  type: (str, Dict[str, str]) -> None

    channels = channels or {}

    hps = {'sagemaker_program': module_name}

    path = os.path.join(DATASETS_PATH, module_name)

    channels = [Channel.create(name, os.path.join(path, file)) for name, file in channels.items()]

    prepare_training_environment(hyperparameters=hps, channels=channels, hosts=hosts)

    script_path = os.path.join(USER_SCRIPTS_PATH, module_name)

    with tmpdir() as tmp:
        code_dir = os.path.join(tmp, 'code_dir')
        shutil.copytree(script_path, code_dir)
        modules.prepare(code_dir, module_name)
        modules.install(code_dir)

        yield

        uninstall(module_name)


def read_json(path):  # type: (str) -> Dict
    with open(path, 'r') as f:
        return json.load(f)


# ##########################FROM SAGEMAKER CONTAINERS TEST###################################
@contextlib.contextmanager
def tmpdir(suffix='', prefix='tmp', dir=None):  # type: (str, str, str) -> None
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    yield tmp
    shutil.rmtree(tmp)


DEFAULT_HYPERPARAMETERS = {'sagemaker_region':                    'us-west-2',
                           'sagemaker_job_name':                  'sagemaker-training-job',
                           'sagemaker_enable_cloudwatch_metrics': False,
                           'sagemaker_container_log_level':       logging.WARNING}


def write_json(obj, path):  # type: (object, str) -> None
    with open(path, 'w') as f:
        json.dump(obj, f)


def create_resource_config(current_host='algo-1', hosts=None):
    # type: (Optional[str], Optional[List]) -> None
    write_json(dict(current_host=current_host, hosts=hosts or ['algo-1']),
               env.resource_config_file_dir)


def create_input_data_config(channels=None):  # type: (Optional[List]) -> None
    channels = channels or []
    input_data_config = {channel.name: channel.config for channel in channels}

    write_json(input_data_config, env.input_data_config_file_dir)


def create_hyperparameters_config(hyperparameters, submit_dir=None, sagemaker_hyperparameters=None):
    # type: (Dict, Optional[str], Optional[Dict]) -> None

    all_hyperparameters = {
        params.SUBMIT_DIR_PARAM: submit_dir or params.DEFAULT_MODULE_NAME_PARAM
    }

    all_hyperparameters.update(sagemaker_hyperparameters or DEFAULT_HYPERPARAMETERS.copy())

    all_hyperparameters.update(hyperparameters)

    write_json(all_hyperparameters, env.hyperparameters_file_dir)


def prepare_training_environment(hyperparameters=None, channels=None,
                                 current_host='algo-1', hosts=None):
    # type: (Optional[Dict], Optional[List], str, Optional[List]) -> None
    hosts = hosts or ['algo-1']
    hyperparameters = hyperparameters or {}

    create_hyperparameters_config(hyperparameters)
    create_resource_config(current_host, hosts)
    create_input_data_config(channels)


DEFAULT_CONFIG = dict(ContentType="application/x-numpy", TrainingInputMode="File",
                      S3DistributionType="FullyReplicated", RecordWrapperType="None")


class Channel(collections.namedtuple('Channel', ['name', 'config'])):
    # type: (str, dict) -> Channel

    @staticmethod
    def create(name, dataset_path=None, config=None):
        # type: (str, Optional[str], Optional[Dict]) -> Channel
        channel = Channel(name, config)
        channel.make_directory()

        if dataset_path:
            shutil.copy2(dataset_path, channel.path)
        return channel

    def make_directory(self):  # type: () -> None
        os.makedirs(self.path)

    @property
    def path(self):  # type: () -> str
        return os.path.join(env._input_data_dir, self.name)
