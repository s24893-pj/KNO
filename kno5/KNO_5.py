import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import keras_tuner

data_names = ["Category", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
              "Flavanoids",
              "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
              "Proline"]

df = pd.read_csv('wine.csv', names=data_names)

one_hot_encoded_data = pd.get_dummies(df, columns=["Category"], dtype="float")

X = one_hot_encoded_data[
    ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
     "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
     "Proline"]]

y = one_hot_encoded_data[["Category_1", "Category_2", "Category_3"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class WineModel(tf.keras.Model):
    def __init__(self, units=32, dropout_rate=0.2):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(units=units, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=units, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout2(x)
        return self.output_layer(x)


def build(hp):
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001])

    model = WineModel(units=units, dropout_rate=dropout_rate)

    model.build(input_shape=(None, 13))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


tuner = keras_tuner.Hyperband(build,
                              objective='val_accuracy',
                              max_epochs=10,
                              factor=3,
                              directory='my_dir',
                              project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Najlepsze parametry: {best_hps.values}")

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

loss, accuracy = hypermodel.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
