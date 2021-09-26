from typing import Iterable

class Perceptron:

    def __init__(self, max_iterations: int = 20, training_multiplier: float = 1, initial_weight: int = 0, bias: int = 0):
        self.max_iterations = max_iterations
        self.training_multiplier = training_multiplier
        self.initial_weight = initial_weight
        self.bias = bias

        self.__fit_completed = False


    def fit(self, training_data: Iterable, training_labels: Iterable):
        if training_data == None:
            raise Exception("No training data provided")

        if training_labels == None:
            raise Exception("No training labels provided")

        if len(training_data) != len(training_labels):
            raise Exception("Training data and labels must be of the same length")

        self.weights = [self.initial_weight] * len(training_data[0])

        for iteration in range(self.max_iterations):
            if iteration != 0:
                if last_weights is self.weights:
                    break

            last_weights = self.weights[:]

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

        return answer


    def __adjust_weights(self, training_example: Iterable, multiplier: int):
        for i in range(len(self.weights)):
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

        return predictions
