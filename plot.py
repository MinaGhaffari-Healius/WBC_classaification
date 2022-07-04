import matplotlib.pyplot as plt
import numpy as np


class model_graphs():
    def draw_learning_curves(self, logfile, epochs):
        plt.close()

        training_acc = []
        training_loss = []
        validation_acc = []
        validation_loss = []
        with open(logfile) as file:
            lines = file.readlines()
        for i in range(epochs):
            training_acc.append(lines[i].split(',')[1])
            training_loss.append(lines[i].split(',')[2])
            validation_acc.append(lines[i].split(',')[3])
            validation_loss.append(lines[i].split(',')[4])
        x = range(1, 42)
        xs, ys = zip(*sorted(zip(x, np.asarray(validation_acc))))
        plt.plot(xs, ys)

        # plt.plot(x, validation_acc)
        plt.show()


if __name__ == "__main__":
    graphs = model_graphs()
    graphs.draw_learning_curves(
        logfile='./training_logfile.txt', epochs=41)
