import numpy as np
import tensorflow as tf

new_model = tf.keras.models.load_model('my_model.keras')
new_model.summary()
def get_user_input():
    print("Wprowadź parametry wina oddzielone przecinkiem")
    user_input = input()
    user_input_list = [float(x) for x in user_input.split(',')]
    return np.array(user_input_list).reshape(1, -1)

def predict_wine_category(model, input_data):
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0] + 1

def predictor(model):
    input_data = get_user_input()

    predicted_category = predict_wine_category(model, input_data)

    print(f"Przewidywana kategoria wina: {predicted_category}")

predictor(new_model)



