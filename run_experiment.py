from argparse import ArgumentParser
import generate_dataset
import sys
import tensorflow as tf
import numpy as np

# These parameter should be passed by cmd line arguments
train_test_ratio = 0.9
N_HIDDEN = 1280
N_INPUT = 188
N_OUTPUT = 88

def process_args():
    # All arguments are processed in this function.
    parsar = ArgumentParser()
    parsar.add_argument("-nn", "--neural_network", dest="neural_network", help="Which neural network architecture to run the experiment", required=True)
    parsar.add_argument("-df", "--data_file", dest="data_file", help="The path to the ground truth file", required=True)
    parsar.add_argument("-o", "--output", dest="output", help="Specify the output file", required=False, default="output.txt")
    parsar.add_argument("-mf", "--model_file", dest="model_file", help="Where to save the word embedding model", required=False, default="word_embedding.model")
    parsar.add_argument("-v", "--verbose", dest="verbose", help="How detail you want the output be", required=False, default="1")
    parsar.add_argument("-we", "--word_embedding", dest="word_embedding", help="Which word embedding model to use", required=False, default="Word2Vec")
    args = parsar.parse_args()
    return args

def build_network(net):
    if net == "FNN":
        tf_model = tf.keras.models.Sequential()

        tf_model.add(tf.keras.layers.Dense(N_HIDDEN, input_shape=(N_INPUT,), activation = 'relu'))
        tf_model.add(tf.keras.layers.Dense(N_HIDDEN, activation='relu'))
        tf_model.add(tf.keras.layers.Dense(N_HIDDEN, activation='relu'))
        tf_model.add(tf.keras.layers.Dense(N_HIDDEN, activation='relu'))
        tf_model.add(tf.keras.layers.Dense(N_OUTPUT, activation='relu'))

        tf_model.summary()

        return tf_model

    elif net == "CNN":
        tf_model = tf.keras.models.Sequential()

        tf_model.add(tf.keras.layers.Convolution1D(32, 4, activation="relu", input_shape=[188,1]))
        tf_model.add(tf.keras.layers.MaxPooling1D(pool_size=4, strides=1))
        # tf_model.add(tf.keras.layers.Convolution1D(16, 4, activation="relu"))
        # tf_model.add(tf.keras.layers.MaxPooling1D(pool_size=4, strides=2))
        tf_model.add(tf.keras.layers.Convolution1D(64, 4, activation="relu"))
        tf_model.add(tf.keras.layers.MaxPooling1D(pool_size=4, strides=4))
        tf_model.add(tf.keras.layers.Flatten())
        tf_model.add(tf.keras.layers.Dense(N_HIDDEN, activation="relu"))
        tf_model.add(tf.keras.layers.Dense(88, activation="relu"))

        tf_model.summary()

        return tf_model

    elif net == "RNN":
        tf_model = tf.keras.models.Sequential()

        tf_model.add(tf.keras.layers.LSTM(128, input_shape=(188, 1)))
        tf_model.add(tf.keras.layers.Dense(N_HIDDEN, activation="relu"))
        tf_model.add(tf.keras.layers.Dense(88, activation="relu"))

        tf_model.summary()
        return tf_model


    else:
        raise RuntimeError("No such network")

        

def main():
    args = process_args()
    neural_network = args.neural_network
    data_file = args.data_file
    output = args.output
    model_file = args.model_file
    verbose = args.verbose
    word_embedding = args.word_embedding

    # Need to erase the argument parsar in the generate_dataset file
    print("try to get dataset")
    dataset = generate_dataset.main(data_file,output,model_file,verbose,word_embedding)
    print("get dataset")

    # build neural network model.
    model = build_network(neural_network)
    model.compile(optimizer = "Adam", loss = "MSE", metrics = ["accuracy"])

    # Split the dataset
    training_data_num = int(len(dataset) * train_test_ratio)

    training = dataset[:training_data_num]
    test = dataset[training_data_num:]

    # Prepare the data into correct shape

    X_train = []
    Y_train = []
    for dat in training:
        X_train.append(dat[0])
        Y_train.append(dat[1])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)


    X_test = []
    Y_test = []

    for dat in test:
        X_test.append(dat[0])
        Y_test.append(dat[1])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    if neural_network == "CNN" or neural_network == "RNN":
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        #Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        #Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], 1))

    print(X_train.shape)
     
    # Train the model
    model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.2)

    # Conduct the experiment
    test_loss, test_acc = model.evaluate(X_test, Y_test)

    print("Test loss: " + str(test_loss) + " Test accuracy: " + str(test_acc))

if __name__ == "__main__":
    main()