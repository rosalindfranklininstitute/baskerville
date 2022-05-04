import tensorflow as tf

@tf.function
def test_eager_tf(x):
    return x @ tf.transpose(x)

x = test_eager_tf(tf.ones((100,1)))
print(f'test eager tf {x.shape} {x.dtype}')

def test_fn():
    yield from range(10)

ds_train = tf.data.Dataset.from_generator(test_fn, output_types=tf.int32, output_shapes=())
print(f'{next(iter(ds_train))}')
