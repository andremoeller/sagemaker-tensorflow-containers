## Folders:

docker/{tf version number}.vanilla/ - Dockerfile with vanilla TF version
docker/{tf version number} - Dockerfile with optimized binaries

Building the container with script mode, pipe mode and num parameter servers:

py2 only for now.

ensure you have the TF binaries in the docker folder:

docker/1.6.0/final/py2/ - tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl
docker/1.7.0/final/py2/ - tensorflow-1.7.0-cp27-cp27mu-linux_x86_64.whl
docker/1.8.0/final/py2/ - tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl

## building the images:

test_build has a parameter version that you can change to build 1.6.0, 1.6.0.vanilla, 1.7.0, 1.7.0.vanilla, 1.8.0, 1.8.0 vanilla

in the end, it will push the image to your ecr account, if you have the repository created
```bash
pip install -U .
pip install .[test]
pytest test/test_builder.py
```

## using pipe mode

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