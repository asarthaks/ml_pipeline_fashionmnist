from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import pickle


# def get_cnn_model(IMAGE_SHAPE) :
#     cnn_model = Sequential([
#         Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = IMAGE_SHAPE),
#         MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
#         Dropout(0.2),
#         Flatten(), # flatten out the layers
#         Dense(32,activation='relu'),
#         Dense(10,activation = 'softmax')
#     ])

#     cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
#     return cnn_model

class CNNModel :
    def __init__(self, trainDataset, out_dir, IMAGE_SHAPE, BATCH_SIZE, TRAIN_EPOCHS):
        """
        Hyper Parameter optimization

        Parameters:
        -----------
        trainDataset: list of data
        out_dir: pathlib.PosixPath
            Path to the output directory
        """
        self.PATH = out_dir
        self.IMAGE_SHAPE = IMAGE_SHAPE
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAIN_EPOCHS = TRAIN_EPOCHS

        self.X_train = trainDataset[0]
        self.y_train = trainDataset[1]
        self.X_validate = trainDataset[2]
        self.y_validate = trainDataset[3]

        

        # self.lgtrain = lgb.Dataset(self.X,label=self.y,
        # 	feature_name=self.colnames,
        # 	categorical_feature = self.categorical_columns,
        # 	free_raw_data=False)

        cnn_model = Sequential([
            Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = IMAGE_SHAPE),
            MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
            Dropout(0.2),
            Flatten(), # flatten out the layers
            Dense(32,activation='relu'),
            Dense(10,activation = 'softmax')
        ])

        cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

        self.model = cnn_model

    def fit(self, model_id=0) :
        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.TRAIN_EPOCHS,
            verbose=1,
            validation_data=(self.X_validate, self.y_validate),
        )
        model_fname = 'model_{}_.h5'.format(model_id)
        self.model.save('{}/{}'.format(self.PATH, model_fname))