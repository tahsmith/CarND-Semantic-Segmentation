import os.path
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
    tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :type sess: tensorflow.Session
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    return (
        sess.graph.get_tensor_by_name(vgg_input_tensor_name),
        sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name),
        sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name),
        sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name),
        sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    )


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    def adapt(input_):
        """
        # 1x1 convs to adapt the classes from the original vgg size to
        # num_classes.
        """
        return tf.layers.conv2d(
            input_, num_classes, 1,
            strides=(1, 1),
            kernel_initializer=tf.initializers.variance_scaling(),
            kernel_regularizer=l2_regularizer(1e-3),
            padding='same'
        )

    def skip(previous, skip_from):
        adapted = adapt(skip_from)
        return tf.add(previous, adapted)

    def upscale(input_, kernel_size, strides):
        return tf.layers.conv2d_transpose(
            input_,
            num_classes,
            kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer=tf.initializers.variance_scaling(mode='fan_out'),
            kernel_regularizer=l2_regularizer(1e-3)
        )

    with tf.variable_scope('fcn8'):
        decoder_input = adapt(vgg_layer7_out)

        upscale_1 = upscale(decoder_input, 4, strides=(2, 2))
        decoder_layer_1 = skip(upscale_1, vgg_layer4_out)

        upscale_2 = upscale(decoder_layer_1, 4, strides=(2, 2))
        decoder_layer_2 = skip(upscale_2, vgg_layer3_out)

        upscale_3 = upscale(decoder_layer_2, 4, strides=(2, 2))
        upscale_4 = upscale(upscale_3, 4, strides=(2, 2))
        upscale_5 = upscale(upscale_4, 4, strides=(2, 2))

        output = upscale_5

    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=nn_last_layer, labels=correct_label)
    cross_entropy = tf.reshape(cross_entropy, [-1, num_classes])
    cost = tf.reduce_mean(cross_entropy)
    regularisation_cost_list = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    for regularisation_cost in regularisation_cost_list:
        cost += regularisation_cost
    optimiser = tf.train.AdamOptimizer(learning_rate)
    fc8_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fcn8')
    kwargs = {}
    if fc8_vars:
        kwargs.update(var_list=fc8_vars)

    train = optimiser.minimize(cost, **kwargs)

    return tf.reshape(nn_last_layer, [-1, num_classes]), train, cost


tests.test_optimize(optimize)


def train_nn(sess: tf.Session, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.initialize_all_variables())

    for epoch_i in range(epochs):
        print('Epoch {i} / {count}'.format(i=epoch_i + 1, count=epochs))
        for image_batch, label_batch in get_batches_fn(batch_size):
            _, cross_entropy_loss_value = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={
                    input_image: image_batch,
                    correct_label: label_batch,
                    keep_prob: 0.7,
                    learning_rate: 1e-3
                }
            )

            print(cross_entropy_loss_value)


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    epochs = 100
    batch_size = 50
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, *vgg_layers = load_vgg(sess, vgg_path)

        fcn8_output = layers(*vgg_layers, num_classes)
        labels = tf.placeholder(tf.float32,
                                [None, image_shape[0], image_shape[1],
                                 num_classes])
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cost = optimize(fcn8_output, labels, learning_rate,
                                          num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cost,
                 input_image, labels, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
