import tensorflow_datasets as tfds
import tensorflow as tf


def get_datasets(num_epochs, batch_size, buffer_size=1024):
    train_ds = tfds.load("mnist", split="train")
    test_ds = tfds.load("mnist", split="test")
    
    train_ds = train_ds.map(lambda sample: {"image": tf.cast(sample["image"], tf.float32) / 255.,
                                            "label": sample["label"]})
    test_ds = test_ds.map(lambda sample: {"image": tf.cast(sample["image"], tf.float32) / 255.,
                                          "label": sample["label"]})
    train_ds = train_ds.repeat(num_epochs).shuffle(buffer_size)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_ds.batch(batch_size).prefetch(1)
    
    return train_ds, test_ds


if __name__ == "__main__":
    trd, ted = get_datasets(2, 32)
    print(trd)
    print(ted)
    pass
    