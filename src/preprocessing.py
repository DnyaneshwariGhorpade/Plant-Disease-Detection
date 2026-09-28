import tensorflow as tf

IMG_SIZE = 96


def decode_and_resize(path, size=IMG_SIZE, to_float=True):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [size, size])
    if to_float:
        img = tf.cast(img, tf.float32) / 255.0
    return img


class DataLoader:
    def __init__(self, paths, labels, num_classes, img_size=IMG_SIZE, batch_size=64, augment=False, shuffle_buffer=4096):
        self.paths = tf.constant(paths)
        self.labels = tf.constant(labels, dtype=tf.int32)
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle_buffer = shuffle_buffer
        self.augmenter = self._build_augmenter() if augment else None

    @staticmethod
    def _build_augmenter():
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])

    def _map_fn(self, path, label):
        img = decode_and_resize(path, self.img_size)
        if self.augmenter is not None:
            img = self.augmenter(img, training=True)
            img = tf.clip_by_value(img, 0.0, 1.0)
        one_hot = tf.one_hot(label, self.num_classes)
        return img, one_hot

    def get_dataset(self, repeat=False):
        ds = tf.data.Dataset.from_tensor_slices((self.paths, self.labels))
        if self.augment:
            ds = ds.shuffle(self.shuffle_buffer)
        ds = ds.map(self._map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        if repeat:
            ds = ds.repeat()
        return ds