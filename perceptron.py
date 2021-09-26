from typing import Iterable
from math import abs


class Perceptron:

    def __init__(self, max_iterations: int = 20, training_multiplier: int = 1, initial_weight: int = 0, bias: int = 0, out_file: str = "output.txt"):
        self.max_iterations = max_iterations
        self.training_multiplier = abs(training_multiplier)
        self.initial_weight = initial_weight
        self.bias = bias

        self.__fit_completed = False
        self.out_file = out_file


    def fit(self, training_data: Iterable, training_labels: Iterable):
        if training_data == None:
            raise Exception("No training data provided")

        if training_labels == None:
            raise Exception("No training labels provided")

        if len(training_data) != len(training_labels):
            raise Exception("Training data and labels must be of the same length")

        self.weights = [self.initial_weight] * len(training_labels)

        for iteration in self.max_iterations:
            if iteration is not 0:
                if last_weights is self.weights:
                    break

            last_weights = self.weights

            for i in range(len(training_data)):
                training_example = training_data[i]

                prediction = self.__predict_with_weight(training_example)

                if prediction > training_labels[i]:
                    self.__adjust_weights(training_example, -1 * self.training_multiplier)
                elif prediction < training_labels[i]:
                    self.__adjust_weights(training_example, self.training_multiplier)

        self.__fit_completed = True


    def __predict_with_weight(self, training_example: Iterable):
        answer = 0

        for i in range(len(training_example)):
            answer += training_example[i] * self.weights[i]

        answer += self.bias

        if answer > 0:
            return 1

        return 0


    def __adjust_weights(self, training_example: Iterable, multiplier: int):
        for i in range(self.weights):
            self.weights[i] += (training_example[i] * multiplier) + self.bias


    def predict(self, testing_data: Iterable, testing_labels: Iterable):
        if not self.__fit_completed:
            raise Exception("Classifier has not been fit to training data")

        if testing_data == None:
            raise Exception("No testing data provided")

        if testing_labels == None:
            raise Exception("No testing labels provided")

        if len(testing_data) != len(testing_labels):
            raise Exception("Testing data and labels must be of the same length")

        predictions = []

        for i in range(len(testing_data)):
            example = testing_data[i]
            predictions.append(self.__predict_with_weight(example))

        self.__write_to_out_file(testing_data, predictions)

        return predictions


    def __write_to_out_file(self, data, predictions):
        if len(data) is not len(predictions):
            raise Exception("Data and predictions must be of the same length")

        with open(self.out_file, "w") as out_file:
            for i in range(len(data)):
                example = data[i]
                prediction = predictions[i]
                out_file.write(str(example) + ":  " + str(prediction))


