from logging import exception
import argparse
import tensorflow as tf

def equations(first_matrix,second_matrix):
    try:
        A1 = tf.constant(first_matrix, dtype=tf.float64)
        A2 = tf.constant(second_matrix, dtype=tf.float64)
        C1 = tf.linalg.inv(A1)
        return tf.matmul(C1, A2)
    except exception as e:
        print("nie ma rozwiązania")


matrix_1 = []
matrix_2 = []

ap = argparse.ArgumentParser()
ap.add_argument("-x", "--aa", nargs='+', required=True,)
ap.add_argument("-z", "--bb", nargs='+', required=True,)
args = vars(ap.parse_args())

for equation in args["aa"]:
    matrix_1.append([int(x) for x in equation.split()])

for result in args["bb"]:
    matrix_2.append([int(x) for x in result.split()])

print(matrix_1)
print(matrix_2)

print(equations(matrix_1, matrix_2))

