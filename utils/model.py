from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam



IMAGE_ROWS = 28
IMAGE_COLS = 28
BATCH_SIZE = 4096
IMAGE_SHAPE = (IMAGE_ROWS, IMAGE_COLS, 1) 


def get_cnn_model(IMAGE_SHAPE) :
    cnn_model = Sequential([
        Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = IMAGE_SHAPE),
        MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
        Dropout(0.2),
        Flatten(), # flatten out the layers
        Dense(32,activation='relu'),
        Dense(10,activation = 'softmax')
    ])

    cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
    return cnn_model