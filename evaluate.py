from data import data_loader
from model import CNN_model
from configuration import config
from keras.models import load_model


if __name__ == '__main__':
    data = data_loader(config['CNN_input_size'], )
    test_X, test_y = data.convert_test_data(config['test_data_path'])
    WBC_CNN_model = CNN_model()
    best_model = load_model('./Models4/model-028.h5')
    # evaluate on blind test set, test_y should be categorical
    # WBC_CNN_model.evaluate_model(best_model, test_X, test_y)
    WBC_CNN_model.plot_CM(best_model, test_X, test_y)  # plot the confusion map
