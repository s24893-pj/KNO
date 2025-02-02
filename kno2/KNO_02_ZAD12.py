import tensorflow as tf
def rotate(x: float, y: float, angle: float) -> tuple:
    _x = tf.constant(x, dtype=tf.float32)
    _y = tf.constant(y, dtype=tf.float32)
    _angle = tf.constant(angle, dtype=tf.float32)
    x1 = tf.subtract((_x * tf.cos(_angle)), (_y * tf.sin(_angle)))
    y1 = tf.add((_y * tf.cos(_angle)), (_x * tf.sin(_angle)))
    return x1, y1

x1, y1 = rotate(5, 0, 45)
print(f"x' and y': {x1}, {y1}")