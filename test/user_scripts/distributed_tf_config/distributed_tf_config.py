import os

if __name__ == '__main__':
    path = os.path.join(os.environ['SM_MODEL_DIR'], os.environ['SM_CURRENT_HOST'])

    print('written to %s' % path)
    with open(path, 'w') as f:
        f.write(os.environ['TF_CONFIG'])
