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
from __future__ import absolute_import, division

import collections
import json
import logging
import os
from threading import Thread

import six
import tensorflow as tf
from typing import Dict, Union, List

from sagemaker_containers.beta import framework

logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Host = collections.namedtuple('Host', 'name num_master num_worker num_ps')


def cluster(current_host_name, hosts, model_dir=None, initial_port=2222):
    """Builds a dictionary containing cluster information based on number of hosts and number of
    parameter servers. More information about TF_Config:

    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn
    /python/learn/estimators/run_config.py#L77 :return: task_type and cluster dictionary
    """
    masters = []
    workers = []
    parameter_servers = []

    is_master = any(host.name == current_host_name and host.num_master for host in hosts)

    task_id = 0
    for host_id, host in enumerate(hosts):
        port = initial_port
        for num, task_type in [(host.num_master, masters),
                               (host.num_worker, workers),
                               (host.num_ps, parameter_servers)]:

            for i in range(num):
                task_name = '%s:%d' % (host.name, port)
                port += 1

                task_type.append(task_name)

                if is_master:
                    task_id = 0
                elif host.name == current_host_name:
                    task_id = host_id - 1

    task_type = 'master' if is_master else 'worker'

    _cluster = {
        'cluster':     {
            'master': masters,
        },
        'task':        {
            'index': task_id,
            'type':  task_type
        },
        'environment': 'cloud',
        'model_dir':   model_dir,
    }

    if parameter_servers:
        _cluster['cluster']['ps'] = parameter_servers

    if workers:
        _cluster['cluster']['worker'] = workers

    return cluster


def run_ps_server(current_host, hosts, cluster):
    def start_ps_server():
        msg = 'starting TF Serving - current host: %s, hosts: %s, cluster %s' % (
            current_host, hosts, cluster)

        logger.info(msg)
        cluster_spec = tf.train.ClusterSpec(cluster)
        task_index = hosts.index(current_host)
        server = tf.train.Server(cluster_spec, job_name='ps', task_index=task_index)
        server.join()

    t = Thread(target=start_ps_server)
    t.start()


def start():
    hyperparameters = framework.env.read_hyperparameters()
    env = framework.training_env(hyperparameters=hyperparameters)

    env_vars = env.to_env_vars()

    model_dir = hyperparameters.get('sagemaker_model_dir', env.model_dir)

    num_ps = hyperparameters.get('sagemaker_num_parameter_servers', None)

    config = tf_config(model_dir, env.current_host, env.hosts, num_ps)

    # TODO(mvs): create an object instead of reading from the dict

    if any(ps.startswith(env.current_host) for ps in config['cluster'].get('ps', [])):
        logger.info('Running PS')
        run_ps_server(env.current_host, env.hosts, config['cluster'])

    os.environ['TF_CONFIG'] = json.dumps(config)

    tf_env_vars = {
        'TF_CONFIG':               json.dumps(config),
        'KMP_SETTINGS':            '1',
        'OMP_NUM_THREADS':         str(env.num_cpus),
        'TF_CPP_MIN_VLOG_LEVEL':   '0',
        'TF_CPP_MIN_LOG_LEVEL':    '3',
        'S3_REQUEST_TIMEOUT_MSEC': '60000',
        'S3_REGION':               os.environ[framework.params.REGION_NAME_ENV],
        'S3_USE_HTTPS':            '0',
        'S3_VERIFY_SSL':           '0',
        'OMP_PROC_BIND':           'true',
        'KMP_AFFINITY':            'granularity=fine,verbose,compact,1,0'
    }

    env_vars.update(tf_env_vars)

    for k, v in hyperparameters.get('sagemaker_env_vars', {}).items():
        env_vars[k] = v

    framework.modules.run_module_from_s3(env.module_dir,
                                         env.to_cmd_args(),
                                         env_vars,
                                         env.module_name)


def default_hosts(current_host, hosts, num_parameters_servers=None):
    if len(hosts) == 1:
        num_ps = num_parameters_servers or 0
        return [Host(name=current_host, num_master=1, num_worker=0, num_ps=num_ps)]
    else:
        num_hosts = len(hosts)
        num_parameters_servers = num_parameters_servers or num_hosts

        ps_per_host = num_parameters_servers // len(hosts)
        mod = num_parameters_servers % len(hosts)

        if mod:
            additional_ps = 1
            mod -= 1
        else:
            additional_ps = 0

        masters = [Host(hosts[0], num_master=1, num_worker=0, num_ps=ps_per_host + additional_ps)]

        workers = []

        for host in hosts[1:]:
            if mod:
                additional_ps = 1
                mod -= 1
            else:
                additional_ps = 0

            host_class = Host(host, num_master=0, num_worker=1, num_ps=ps_per_host + additional_ps)
            workers.append(host_class)

        return masters + workers


def tf_config(model_dir, current_host, hosts, num_parameters_servers=None):
    # type: (str, str, Union[List[Host], List[str]]) -> Dict[str, str]

    is_hostname = isinstance(hosts[0], six.string_types)
    hosts = default_hosts(current_host, hosts, num_parameters_servers) if is_hostname else hosts

    return cluster(current_host_name=current_host, hosts=hosts, model_dir=model_dir)
