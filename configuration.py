config = dict()

config['train_data_path'] = './dataset2-master/dataset2-master/images/TRAIN/'
config['test_data_path'] = './dataset2-master/dataset2-master/images/Test/'
config['CNN_input_size'] = (80, 60)
# config['CNN_input_size'] = (180, 180)
config['validation_percentage'] = 0.2

config['cnn_input_shape'] = (60, 80, 3)
# config['cnn_input_shape'] = (180, 180, 3)
config['n_epochs'] = 100
config['batch_size'] = 64
config['early_stopping_patience'] = 20
config['logfile_name'] = "./training_logfile.txt"
