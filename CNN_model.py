#importing necessary libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
import pandas as pd


def CNN_model():
# seed for reproducing same results
    seed = 42
    np.random.seed(seed)

    # loading dataset
    data = pd.read_csv('image_data.csv')

    # splitting into input and output variables
    X = []
    Y = data['y']
    del data['y']
    del data['Character']
    for i in range(data.shape[0]):
        flat_pixels = data.iloc[i].values[1:]
        image = np.reshape(flat_pixels, (28,28)) #28x28
        X.append(image)

    X = np.array(X) #conversion to numpy arrays
    Y = np.array(Y)

    # split the dataset in a 50-50 configuration
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.30, random_state=seed)

    # one hot encoding  outputs because they are categorical data
    Y_test_for_accuracy_matrix = Y_test.copy()
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)

    #reshaping data
    X_train = X_train.reshape(-1,28,28,1)
    X_test  = X_test.reshape(-1,28,28,1)

    num_classes = 37

  

    #model creation
    model_ = Sequential()
    model_.add(Conv2D(32, (24,24), input_shape=(28, 28, 1), activation='relu'))
    model_.add(MaxPooling2D(pool_size=(2, 2)))
    model_.add(Dropout(0.4))
    model_.add(Flatten())
    model_.add(Dense(128, activation='relu'))
    #output layer
    model_.add(Dense(36, activation='softmax'))


    # Compiling model
    model_.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=200, verbose=2)

  
    from sklearn.metrics import accuracy_score
    y_pred = model_.predict_classes(X_test)
    acc = accuracy_score(Y_test_for_accuracy_matrix, y_pred)
    print(f"Accuracy: {acc}")
    
    return model_
    
