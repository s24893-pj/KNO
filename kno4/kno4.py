import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
df = pd.read_csv("wine.csv")

df.reset_index()

data_names = ["Category","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
              "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
              "Proline"]

df.columns = data_names

df = df.sample(frac = 1)

one_hot_encoded_data = pd.get_dummies(df, columns=["Category"], dtype="float")

print(one_hot_encoded_data)


X = one_hot_encoded_data[["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
              "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
              "Proline"]]

y = one_hot_encoded_data[["Category_1", "Category_2", "Category_3"]]

X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size=0.3, random_state=1)

X_val, X_test, y_val, y_test = train_test_split(X_test_, y_test_, test_size=0.5, random_state=1)

baseline_model = tf.keras.models.load_model('my_model2.keras')

import tensorflow as tf


def create_model(dropout_rate, units, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=units[0], activation='relu'))

    for units in units[1:]:
        model.add(tf.keras.layers.Dense(units=units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy,
                metrics=['accuracy'])

    return model


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def test_model():
    dropout_rates = [0.2, 0.4]
    units_list = [[24, 36, 12,64,55], [128, 64, 32], [64, 128, 64,100,95]]
    learning_rates = [0.05, 0.01]

    results = []

    for dropout_rate in dropout_rates:
        for units in units_list:
            for learning_rate in learning_rates:
                print(f"Trening modelu z dropout_rate={dropout_rate} units={units} learning_rate={learning_rate}")

                model = create_model(dropout_rate=dropout_rate, units=units, learning_rate=learning_rate)

                model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

                test_loss, test_accuracy = model.evaluate(X_test, y_test)

                results.append({
                    'dropout_rate': dropout_rate,
                    'units': units,
                    'learning_rate': learning_rate,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy
                })

    results_df = pd.DataFrame(results)

    best_model = results_df.sort_values(by='test_accuracy', ascending=False).iloc[0]

    return results_df, best_model


results_df, best_model = test_model()

print("\nWyniki:")
print(results_df)

print("\nNajlepszy model:")
print(best_model)

baseline_test_loss, baseline_test_accuracy = baseline_model.evaluate(X_test, y_test, verbose=0)

print("baseline - Test loss:", baseline_test_loss)
print("baseline - Test accuracy:", baseline_test_accuracy)
