from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation


def build_cnn_model(classes_number, input_shape):
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Dense(classes_number, activation='softmax'))
	model.compile(
		loss='categorical_crossentropy',
		optimizer="adam",
		metrics=['accuracy']
	)
	model.summary()
	return model
