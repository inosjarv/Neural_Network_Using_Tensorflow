from preprocessing_data import *
from nn_tf import *
if __name__ == "__main__":
    csv_file = DataProcessor(filename='data.csv')
    csv_file.run()

    nn = NeuralNetwork('input.csv')
    nn.run()

