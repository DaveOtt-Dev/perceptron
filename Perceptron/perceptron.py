from typing import Iterable, Literal

class Perceptron:

    def __init__(self, max_iterations: int = 1, training_multiplier: float = 1, initial_weight: int = 0, bias: int = 0, averaged: bool = False):
        """Perceptron

        Perceptron Initializer

        Args:
            max_iterations: Number of iterations through the data completed during fit.
              Default: 1
            training_multiplier: Weights are updated when an incorrect prediction is made
              during fitting. Multiplier determines what to multiply the data by when making
              this update.
              Default: 1
            initial_weight: Weights are initialized to this number before fitting.
              Default: 0
            bias: Training bias. Bias is added in to the predicted value
              Default: 0
            averaged: Set to true if you want to make use of an averaged perceptron rather than
              the traditional
              Default: False
        """
        self.max_iterations = max_iterations
        self.training_multiplier = training_multiplier
        self.initial_weight = initial_weight
        self.bias = bias

        self.__fit_completed = False
        self.averaged = averaged


    def fit(self, training_data: Iterable, training_labels: Iterable) -> None:
        """
        Fits the perceptron model to the provided training data and labels.

        This method trains the perceptron by iterating through the training data and updating
        the weights based on the predictions and actual labels. It supports both standard and
        averaged perceptron training, depending on the `averaged` attribute of the perceptron.

        Args:
            training_data (Iterable): The training data, where each element is a feature vector.
            training_labels (Iterable): The corresponding labels for the training data.

        Raises:
            Exception: If `training_data` is None.
            Exception: If `training_labels` is None.
            Exception: If the lengths of `training_data` and `training_labels` do not match.
        """
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
        """
        Trains the perceptron using the standard (non-averaged) training method.

        This method iterates through the training data for a specified number of iterations,
        updating the weights whenever a prediction does not match the actual label. The weights
        are adjusted based on the training multiplier and the feature vector of the misclassified
        example.

        Args:
            training_data (Iterable): The training data, where each element is a feature vector.
            training_labels (Iterable): The corresponding labels for the training data.

        Raises:
            Exception: If the training data or labels are invalid (handled by the calling method).
        """
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
        """
        Trains the perceptron using the averaged training method.

        This method iterates through the training data for a specified number of iterations,
        updating the weights whenever a prediction does not match the actual label. The weights
        are adjusted based on the training multiplier and the feature vector of the misclassified
        example. After all iterations, the final weights are computed as the average of all
        intermediate weights.

        Args:
            training_data (Iterable): The training data, where each element is a feature vector.
            training_labels (Iterable): The corresponding labels for the training data.

        Raises:
            Exception: If the training data or labels are invalid (handled by the calling method).
        """
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
        """
        Computes the averaged weights from a list of weight vectors.

        This method calculates the average value for each weight across all iterations
        and returns the resulting averaged weight vector.

        Args:
            weights (Iterable): A list of weight vectors, where each vector corresponds
                                to the weights at a specific iteration.

        Returns:
            Iterable: The averaged weight vector, where each element is the mean of the
                    corresponding weights across all iterations.
        """
        averaged_weights = []

        for i in range(len(weights[0])):
            sum = 0
            for weight in weights:
                sum += weight[i]
            
            averaged_weights.append(sum / len(weights))

        return averaged_weights


    def __predict_with_weight(self, training_example: Iterable, weights: Iterable) -> Literal[1, 0]:
        """
        Predicts the label for a given training example using the provided weights.

        This method calculates the weighted sum of the feature values and adds the bias.
        If the result is greater than 0, it predicts 1; otherwise, it predicts 0.

        Args:
            training_example (Iterable): A feature vector representing the training example.
            weights (Iterable): The weight vector used for prediction.

        Returns:
            Literal[1, 0]: The predicted label (1 or 0) for the given training example.
        """
        answer = 0

        for i in range(len(training_example)):
            answer += training_example[i] * weights[i]

        answer += self.bias

        if answer > 0:
            return 1

        return 0


    def __adjust_weights(self, weights: Iterable, training_example: Iterable, multiplier: int) -> Iterable:
        """
        Adjusts the weights based on a training example and a multiplier.

        This method updates each weight by adding the product of the corresponding feature
        value in the training example and the multiplier, along with the bias.

        Args:
            weights (Iterable): The current weight vector to be updated.
            training_example (Iterable): A feature vector representing the training example.
            multiplier (int): A value that determines the direction and magnitude of the adjustment.

        Returns:
            Iterable: The updated weight vector after applying the adjustments.
        """
        for i in range(len(weights)):
            weights[i] += (training_example[i] * multiplier) + self.bias

        return weights


    def predict(self, testing_data: Iterable, testing_labels: Iterable) -> Iterable:
        """
        Predicts the labels for the given testing data using the trained perceptron model.

        This method uses the trained weights to predict the labels for each example in the
        testing data. It ensures that the perceptron has been trained before making predictions
        and validates the input data and labels.

        Args:
            testing_data (Iterable): The testing data, where each element is a feature vector.
            testing_labels (Iterable): The corresponding labels for the testing data.

        Raises:
            Exception: If the perceptron has not been trained.
            Exception: If `testing_data` is None.
            Exception: If `testing_labels` is None.
            Exception: If the lengths of `testing_data` and `testing_labels` do not match.

        Returns:
            Iterable: A list of predicted labels for the testing data.
        """
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
