import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from config import config


class EncodingBlock(tf.keras.Model):
    def __init__(self, num_filters, input_shape=None, is_pool=True, is_relu=True, conv_filter_size=3, strides=1,
                 padding='same'):
        super(EncodingBlock, self).__init__()
        if (input_shape != None):
            self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=conv_filter_size,
                                               input_shape=input_shape, padding=padding, strides=strides)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=conv_filter_size, padding=padding,
                                               strides=strides)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu = tf.keras.layers.ReLU() if is_relu else None
        self.pool = tf.keras.layers.MaxPool2D((2, 2)) if is_pool else None

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training)
        x = self.relu(x) if self.relu != None else x
        return self.pool(x) if self.pool != None else x


class EncodingNet(tf.keras.Model):
    def __init__(self, input_shape, num_blocks, num_filters, pooling=-1):
        super(EncodingNet, self).__init__()
        if (pooling == -1):
            pooling = [True for i in range(num_blocks)]
        self.blocks = [0 for i in range(num_blocks)]
        self.blocks[0] = EncodingBlock(num_filters[0], input_shape=input_shape, is_pool=pooling[0])
        for i in range(1, num_blocks):
            self.blocks[i] = EncodingBlock(num_filters[i], is_pool=pooling[i])

    def call(self, x, training=False):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training)
        return x


class GLAMORNet(tf.keras.Model):
    def __init__(self, num_classes=7, face_input_shape=(96, 96, 3), context_input_shape=(112, 112, 3)):
        super(GLAMORNet, self).__init__()
        self.FaceEncodingNet = EncodingNet(input_shape=face_input_shape,
                                           num_blocks=config.face_encoding.num_blocks,
                                           num_filters=config.face_encoding.num_filters,
                                           pooling=config.face_encoding.pooling)

        self.ContextEncodingNet = EncodingNet(input_shape=context_input_shape,
                                              num_blocks=config.context_encoding.num_blocks,
                                              num_filters=config.context_encoding.num_filters,
                                              pooling=config.context_encoding.pooling)

        self.FaceReduction = tf.keras.layers.GlobalAveragePooling2D()  # convert the encoded face tensor to a single vector

        # GLA module
        self.attention_fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.attention_fc1_bn = tf.keras.layers.BatchNormalization()
        self.attention_fc2 = tf.keras.layers.Dense(units=1, activation=None)
        self.attention_dot = tf.keras.layers.Dot(axes=1)

        # Fusion module
        self.face_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.face_weight2 = tf.keras.layers.Dense(units=1, activation=None)
        self.context_weight1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.context_weight2 = tf.keras.layers.Dense(units=1, activation=None)
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)
        self.softmax1 = tf.keras.layers.Activation('softmax')

        # LSTM layer
        self.lstm = tf.keras.layers.LSTM(units=128, return_sequences=True)

        # Classifier after fusion
        self.final_fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.final_dropout1 = tf.keras.layers.Dropout(rate=config.dropout_rate)
        self.final_classify = tf.keras.layers.Dense(units=num_classes, activation='softmax')

def call(self, x_face, x_context, training=False):
    face_dim, context_dim = 3, 3  # Assuming x_face and x_context are both 3-dimensional

    face_vectors = []
    context_vectors = []

    for dim in range(face_dim):
        face = self.FaceEncodingNet(x_face[dim], training=training)
        context = self.ContextEncodingNet(x_context[dim], training=training)

        face_vector = self.FaceReduction(face)  # dim [1xC]
        face_vectors.append(face_vector)
        context_vectors.append(context)

    # GLA module
    concat1_list = []
    attention_weight_list = []
    for dim in range(face_dim):
        N, H, W, C = context_vectors[dim].shape
        face_vector_repeat = tf.keras.layers.RepeatVector(H * W)(face_vectors[dim])
        context_vector = tf.keras.layers.Reshape((H * W, C))(context_vectors[dim])
        concat1 = tf.keras.layers.Concatenate(axis=-1)([face_vector_repeat, context_vector])
        concat1_list.append(concat1)

        attention_weight = self.attention_fc1(concat1)
        attention_weight = self.attention_fc1_bn(attention_weight, training=training)
        attention_weight = tf.keras.layers.Activation("relu")(attention_weight)
        attention_weight = self.attention_fc2(attention_weight)
        attention_weight = tf.nn.softmax(attention_weight, axis=1)
        attention_weight_list.append(attention_weight)

    context_vectors_after_attention = []
    for dim in range(face_dim):
        context_vector = context_vectors[dim]
        attention_weight = attention_weight_list[dim]
        context_vector = self.attention_dot([context_vector, attention_weight])
        context_vector = tf.keras.layers.Reshape((C,))(context_vector)
        context_vectors_after_attention.append(context_vector)

    w_f_list = [self.face_weight1(face_vectors[dim]) for dim in range(face_dim)]
    w_f_list = [self.face_weight2(w_f) for w_f in w_f_list]
    w_c_list = [self.context_weight1(context_vectors_after_attention[dim]) for dim in range(face_dim)]
    w_c_list = [self.context_weight2(w_c) for w_c in w_c_list]

    w_fc_list = [self.concat1([w_f_list[dim], w_c_list[dim]]) for dim in range(face_dim)]
    w_fc_list = [self.softmax1(w_fc) for w_fc in w_fc_list]

    face_vectors = [face_vectors[dim] * tf.expand_dims(w_fc[:, 0], axis=-1) for dim, w_fc in enumerate(w_fc_list)]
    context_vectors_after_attention = [context_vectors_after_attention[dim] * tf.expand_dims(w_fc[:, 1], axis=-1) for dim, w_fc in enumerate(w_fc_list)]

    concat2_list = [tf.keras.layers.Concatenate(axis=-1)([face_vectors[dim], context_vectors_after_attention[dim]]) for dim in range(face_dim)]

    # Combine the results from all dimensions
    concat2 = tf.keras.layers.Concatenate(axis=-1)(concat2_list)

    # TODO: Add temporal information
    # Combine the results from all dimensions
    concat2 = tf.keras.layers.Concatenate(axis=-1)(concat2_list)

    # Add LSTM layer with sequence length of 3
    lstm1 = self.lstm(tf.keras.layers.Reshape((3, -1))(concat2))

    features = self.final_fc1(lstm1)
    features = self.final_dropout1(features, training=training)
    scores = self.final_classify(features)

    return scores



def get_model():
    model = GLAMORNet(config.num_classes, config.face_input_size + [3], config.context_input_size + [3])
    #model.call(tf.keras.Input(config.face_input_size + [3]), tf.keras.Input(config.context_input_size + [3])) #build the model
    return model

if __name__ == '__main__':
    a = tf.random.normal([2, 96, 96, 3])
    b = tf.random.normal([2, 112, 112, 3])
    print(config.train_images)
    model = get_model()


    model.call(tf.keras.Input((96, 96, 3)), tf.keras.Input((112, 112, 3)))
    o = model(a, b, True)
    print(model.summary())
    print(o.shape)
