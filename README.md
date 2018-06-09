# Overview

This branch contains a number of work-in-progress features for SageMaker TensorFlow,
including:

* MKL in CPU instances
* Pipe mode support (for TF 1.7)
* Script mode (run a TensorFlow script as an ordinary Python script rather than implementing various functions)
* SSL / HTTPS toggle when downloading data from S3.

## How do I run these containers?

Using the Python SDK, running this code from the `test/integration/benchmarks/criteo` directory will
runs a training job with 2 ml.c5.9xlarge instances with 2 parameter servers.

```python

from sagemaker.tensorflow import TensorFlow

# Change this to your own criteo data bucket
DATASETS_CRITEO_SMALLCLICKS = "s3://sagemaker-us-west-2-369233609183/datasets/criteo-dataset"

# sagemaker_num_parameter_servers sets the number of parameter servers in the cluster.
hps = {
    'sagemaker_num_parameter_servers': 2,
	's3_channel':                   DATASETS_CRITEO_SMALLCLICKS,
    'batch_size':                   30000,
    'dataset':                      'kaggle',
    'model_type':                   'linear',
    'l2_regularization':            100,
    'inter_op_parallelism_threads': 10,
    'sagemaker_env_vars':           {
        'S3_USE_HTTPS':            True,
        'S3_VERIFY_SSL':           True,
        'KMP_BLOCKTIME':           25
    }
}

#pass in your own role
tf = TensorFlow(entry_point='task.py',
                source_dir='trainer',
                train_instance_count=2,
                train_instance_type='ml.c5.9xlarge',
                role='SageMakerRole',
                hyperparameters=hps)

# This points to the prototype images.
# Change the region (to us-west-2 or us-east-2) or TF version (to 1.7.0) if needed
tf.train_image = lambda: '520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tensorflow:1.6.0-cpu-py2-script-mode-preview'

# publicly accessible placeholder data. Change the region if needed
tf.fit({'training':'s3://sagemaker-sample-data-us-east-1/spark/mnist/train'})

```

## Developer documentation (WIP)

### Directories:

docker/{tf version number}.vanilla/ - Dockerfile with vanilla TF version
docker/{tf version number} - Dockerfile with optimized binaries

Building the container with script mode, pipe mode and num parameter servers:

py2 only for now.

ensure you have the TF binaries in the docker folder:

docker/1.6.0/final/py2/ - tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl
docker/1.7.0/final/py2/ - tensorflow-1.7.0-cp27-cp27mu-linux_x86_64.whl
docker/1.8.0/final/py2/ - tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl

### Building the Images:

test_build has a parameter version that you can change to build 1.6.0, 1.6.0.vanilla, 1.7.0, 1.7.0.vanilla, 1.8.0, 1.8.0 vanilla

in the end, it will push the image to your ecr account, if you have the repository created
```bash
pip install -U .
pip install .[test]
pytest test/test_builder.py
```

### Using Pipe Mode

The 1.7.0 and 1.8.0 images include pipe mode installed, example of usage:

```python
features = {
    'data': tf.FixedLenFeature([], tf.string),
    'labels': tf.FixedLenFeature([], tf.int64),
}

def parse(record):
    parsed = tf.parse_single_example(record, features)
    return ({
        'data': tf.decode_raw(parsed['data'], tf.float64)
    }, parsed['labels'])

ds = PipeModeDataset(channel='training', record_format='TFRecord')
num_epochs = 20
ds = ds.repeat(num_epochs)
ds = ds.prefetch(10)
ds = ds.map(parse, num_parallel_calls=10)
ds = ds.batch(64)
```

More information about pipe mode https://github.com/aws/sagemaker-tensorflow-extensions