from typing import Iterable, Literal

class Perceptron:

    def __init__(self, max_iterations: int = 1, training_multiplier: float = 1, initial_weight: int = 0, bias: int = 0, averaged: bool = False):
        self.max_iterations = max_iterations
        self.training_multiplier = training_multiplier
        self.initial_weight = initial_weight
        self.bias = bias

        self.__fit_completed = False
        self.averaged = averaged


    def fit(self, training_data: Iterable, training_labels: Iterable) -> None:
        if training_data == None:
            raise Exception("No training data provided")

        if training_labels == None:
            raise Exception("No training labels provided")

        if len(training_data) != len(training_labels):
            raise Exception("Training data and labels must be of the same length")

        if self.averaged:
            self.__fit_averaged(training_data, training_labels)
        else:
            self.__fit_normal(training_data, training_labels)

        self.__fit_completed = True


    def __fit_normal(self, training_data: Iterable, training_labels: Iterable) -> None:
        weights = [self.initial_weight] * len(training_data[0])

        for iteration in range(self.max_iterations):
            if iteration != 0:
                if last_weights is weights:
                    break

            last_weights = weights[:]

            for i in range(len(training_data)):
                training_example = training_data[i]

                prediction = self.__predict_with_weight(training_example, weights)

                if prediction > training_labels[i]:
                    weights = self.__adjust_weights(weights, training_example, -1 * self.training_multiplier)
                elif prediction < training_labels[i]:
                    weights = self.__adjust_weights(weights, training_example, self.training_multiplier)

        self.weights = weights


    def __fit_averaged(self, training_data: Iterable, training_labels: Iterable) -> None:
        weights = []
        current_weight = [self.initial_weight] * len(training_data[0])

        for iteration in range(self.max_iterations):
            for i in range(len(training_data)):
                training_example = training_data[i]

                prediction = self.__predict_with_weight(training_example, current_weight)

                if prediction > training_labels[i]:
                    current_weight = self.__adjust_weights(current_weight, training_example, -1 * self.training_multiplier)
                elif prediction < training_labels[i]:
                    current_weight = self.__adjust_weights(current_weight, training_example, self.training_multiplier)

                weights.append(current_weight[:])

        self.weights = self.__get_averaged_weights(weights)

        
    def __get_averaged_weights(self, weights: Iterable) -> Iterable:
        averaged_weights = []

        for i in range(len(weights[0])):
            sum = 0
            for weight in weights:
                sum += weight[i]
            
            averaged_weights.append(sum / len(weights))

        return averaged_weights


    def __predict_with_weight(self, training_example: Iterable, weights: Iterable) -> Literal[1, 0]:
        answer = 0

        for i in range(len(training_example)):
            answer += training_example[i] * weights[i]

        answer += self.bias

        if answer > 0:
            return 1

        return 0


    def __adjust_weights(self, weights: Iterable, training_example: Iterable, multiplier: int) -> Iterable:
        for i in range(len(weights)):
            weights[i] += (training_example[i] * multiplier) + self.bias

        return weights


    def predict(self, testing_data: Iterable, testing_labels: Iterable) -> Iterable:
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
            predictions.append(self.__predict_with_weight(example, self.weights))

        return predictions
