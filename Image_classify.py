import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
import matplotlib.image as mimage
# import matplotlib.pyplot as plot
# from PIL import Image


class Image_classify:
    # the file path of weight
    network_weight_filePath = "static/network_weight"
    model = object
    train_model = False

    # function to process data before training
    def process_data(a, b):
        a = tf.cast(a, tf.float32) / 255.0  # cast to 0-1.0
        a = tf.expand_dims(a, axis=2)  # reshape x from [6000,28,28] to [6000,28,28,1]
        b = tf.one_hot(b, depth=10)
        return a, b

    def __init__(self):

        # set config param
        train_batch_size = 1000
        validate_batch_size = 1000
        epoch = 10
        learning_rate = 0.01
        network_weight_filePath = self.network_weight_filePath
        train_model = self.train_model  # train or not

        # get data from mnist & process data & set optimizer
        (data_train, data_train_val), (data_validate, data_validate_val) = datasets.mnist.load_data()
        train_db = tf.data.Dataset.from_tensor_slices((data_train, data_train_val))  # construct dataset object
        train_db = train_db.map(Image_classify.process_data)
        train_db = train_db.shuffle(10000)
        train_db = train_db.batch(batch_size=train_batch_size)
        validate_db = tf.data.Dataset.from_tensor_slices((data_validate, data_validate_val))  # construct dataset object
        validate_db = validate_db.map(Image_classify.process_data)
        validate_db = validate_db.shuffle(5000)
        validate_db = validate_db.batch(validate_batch_size)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # model input shape [batch_size,28,28,1]
        model = keras.Sequential([
            layers.Conv2D(6, kernel_size=3, strides=1),
            layers.ReLU(),
            layers.Conv2D(6, kernel_size=3),
            layers.ReLU(),
            layers.Conv2D(6, kernel_size=3),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(56, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(28, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10),
            layers.Softmax()
        ])
        self.model = model

        # load weights from file
        try:
            model.load_weights(network_weight_filePath)
            print(f"network load weight from {network_weight_filePath}")
        except Exception:
            print(f"network weight file not found file path {network_weight_filePath}")

        # train
        if train_model:
            # training in epoch times
            for epoch_t in range(epoch):
                # one epoch for all training data
                for step, (x, y) in enumerate(train_db):
                    # if step == 3:
                    #     break
                    with tf.GradientTape() as tape:
                        out = model(x)
                        loss = tf.losses.categorical_crossentropy(y_true=y, y_pred=out)
                        loss = tf.reduce_sum(loss)
                        grads = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                        print(f"epoch {epoch_t} step {step} loss {loss}")

                # save weights to the file
                model.save_weights(network_weight_filePath)
                print(f"model save network weight to {network_weight_filePath}")

                # statistic the accarucy of model from validation data
                (x, y) = next(iter(validate_db))
                y_pre = model.predict(x)
                y_pre = tf.argmax(y_pre, axis=1)
                y = tf.argmax(y, axis=1)
                accuracy = tf.cast(tf.equal(y, y_pre), tf.float32)
                accuracy = tf.reduce_sum(accuracy) / accuracy.shape[0]
                print(f"epoch {epoch_t} accuracy {accuracy}")

    # process receive image to a proper shape
    def process_receive_image(self, image):
        img = mimage.imread(image)  # shape [500,500,4]
        img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, [28, 28])  # shape [28,28,4]
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2] +img[:, :, 3]  # shape [28,28]
        img = tf.expand_dims(img, axis=2)  # shape [28,28,1]
        # plot.imshow(img)
        # plot.show()
        img = tf.expand_dims(img, axis=0)  # shape [1,28,28,1]
        print("image pre process ")
        return img

    def recognise(self, image):
        x = self.process_receive_image(image)
        # x = tf.random.normal([1, 28, 28, 1])# create x use for development test
        y_pre = self.model(x)  # x shape shoul be [1,28,28,1]
        y_pre = y_pre[0]
        y_pre = tf.argmax(y_pre)
        y_pre = tf.get_static_value(y_pre)  # cast tensor to python number
        print(f"test image recognise {y_pre}")
        return y_pre

# claasify = Image_classify()
# claasify.recognise()
