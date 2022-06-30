from configuration import config
from data import data_loader
from model import CNN_model


if __name__ == '__main__':

    data = data_loader(config['CNN_input_size'], )
    train_X, train_y, val_X, val_y = data.convert_training_data(
        config['train_data_path'], config['validation_percentage'])

    WBC_CNN_model = CNN_model()
    WBC_model = WBC_CNN_model.define_model()
    # WBC_model = WBC_CNN_model.resnet_model()
    WBC_CNN_model.train_model(WBC_model, train_X, train_y, val_X, val_y,
                              config['n_epochs'], config['batch_size'], config['early_stopping_patience'], config['logfile_name'])
