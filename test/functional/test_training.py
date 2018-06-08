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

import json
import os

from sagemaker_containers.beta import framework

from sagemaker_tensorflow_container import training
import test


def test_single_host_training():
    with test.prepare_training('iris', channels={
        'training': 'iris_training.csv',
        'test':     'iris_test.csv'
    }):
        env = framework.training_env()
        env_vars = env.to_env_vars()

        config = training.tf_config(env.model_dir, env.current_host, env.hosts)

        env_vars['TF_CONFIG'] = json.dumps(config)

        framework.modules.write_env_vars(env_vars)
        framework.modules.run(module_name='iris', env_vars=env_vars)

        assert os.path.exists(os.path.join(env.model_dir, 'checkpoint'))
        assert os.path.exists(os.path.join(env.model_dir, 'graph.pbtxt'))
        assert os.path.exists(os.path.join(env.model_dir, 'eval'))


def test_distributed_tf_config():
    hosts = ['algo-1', 'algo-2', 'algo-3']

    with test.prepare_training('distributed_tf_config', hosts=hosts):
        env = framework.training_env()

        env_vars_list = []
        model_dir = env.model_dir
        for host in hosts:
            env._current_host = host

            env_vars = env.to_env_vars()
            config = training.tf_config(model_dir, host, hosts)
            env_vars['TF_CONFIG'] = json.dumps(config)

            env_vars_list.append(env_vars)

        framework.modules.run_distributed(module_name='distributed_tf_config', num_instances=3,
                                          env_vars_list=env_vars_list)

        common_config = {
            'cluster':     {
                'master': ['algo-1:2222'],
                'worker': ['algo-2:2222', 'algo-3:2222'],
                'ps':     ['algo-1:2223', 'algo-2:2223', 'algo-3:2223']
            },
            'environment': 'cloud',
            'model_dir':   model_dir,
        }

        algo_1 = {'task': {'index': 0, 'type': 'master'}}
        algo_1.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-1')) == algo_1

        algo_2 = {'task': {'index': 1, 'type': 'worker'}}
        algo_2.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-2')) == algo_2

        algo_3 = {'task': {'index': 2, 'type': 'worker'}}
        algo_3.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-3')) == algo_3


def test_distributed_tf_config_with_custom_ps():
    hosts = ['algo-1', 'algo-2', 'algo-3', 'algo-4', 'algo-5', 'algo-6', 'algo-7', 'algo-8']

    with test.prepare_training('distributed_tf_config', hosts=hosts):
        env = framework.training_env()

        env_vars_list = []
        model_dir = env.model_dir
        for host in hosts:
            env._current_host = host

            env_vars = env.to_env_vars()
            config = training.tf_config(model_dir, host, hosts, num_parameters_servers=4)
            env_vars['TF_CONFIG'] = json.dumps(config)

            env_vars_list.append(env_vars)

        framework.modules.run_distributed(module_name='distributed_tf_config', num_instances=3,
                                          env_vars_list=env_vars_list)

        common_config = {
            'cluster':     {
                'master': ['algo-1:2222'],
                'worker': ['algo-2:2222', 'algo-3:2222', 'algo-4:2222', 'algo-5:2222',
                           'algo-6:2222', 'algo-7:2222', 'algo-8:2222'],
                'ps':     ['algo-1:2223', 'algo-2:2223', 'algo-3:2223', 'algo-4:2223']
            },
            'environment': 'cloud',
            'model_dir':   model_dir,
        }

        algo_1 = {'task': {'index': 0, 'type': 'master'}}
        algo_1.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-1')) == algo_1

        algo_2 = {'task': {'index': 1, 'type': 'worker'}}
        algo_2.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-2')) == algo_2

        algo_3 = {'task': {'index': 2, 'type': 'worker'}}
        algo_3.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-3')) == algo_3

        algo_4 = {'task': {'index': 3, 'type': 'worker'}}
        algo_4.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-4')) == algo_4

        algo_5 = {'task': {'index': 4, 'type': 'worker'}}
        algo_5.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-5')) == algo_5

        algo_6 = {'task': {'index': 5, 'type': 'worker'}}
        algo_6.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-6')) == algo_6

        algo_7 = {'task': {'index': 6, 'type': 'worker'}}
        algo_7.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-7')) == algo_7

        algo_8 = {'task': {'index': 7, 'type': 'worker'}}
        algo_8.update(common_config)

        assert test.read_json(os.path.join(model_dir, 'algo-8')) == algo_8
