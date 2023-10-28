class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().
    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


base_config = Config({
    'name': 'Base Config',

    # Training images and cropped faces
    'train_images': 'temporal-data/NewData2/Train',
    'train_crop': 'temporal-data/NewData2/Output_Train',

    # Valid images and cropped faces
    'val_images': 'temporal-data/NewData2/Validation',
    'val_crop': 'temporal-data/NewData2/Output_Validation',

    # test images and cropped faces
    'test_images': 'temporal-data/NewData2/Test',
    'test_crop': 'temporal-data/NewData2/Output_Test',

    'train_label_csv': 'temporal-data-labels/TrainLabels.csv',
    'test_label_csv': 'temporal-data-labels/TestLabels.csv',
    'val_label_csv': 'temporal-data-labels/ValidationLabels.csv',
    'label_column': 2,

    'face_input_size': [96, 96],
    'context_input_size': [112, 112],
    'batch_size': 4,
    'num_parallel_calls' : 2,

    'lr': 5e-4,  # initial learning rate
    'momentum': 0.9,

    'lr_decay': 0.1,
    'lr_steps': [25, 40],  # after some interval (epochs), decay the lr by multiplying with lr_decay
    'lr_minbound' : 1e-5, #minimum possible learning rate

    'face_encoding': Config({'num_blocks': 5, 'num_filters': [32, 64, 128, 256, 256], 'pooling': [1, 1, 1, 1, 0]}),
    # pooling except the last layer
    'context_encoding': Config({'num_blocks': 5, 'num_filters': [32, 64, 128, 256, 256], 'pooling': [1, 1, 1, 1, 0]}),
    # pooling except the last layer

    'dropout_rate': 0.5,
    'num_classes': 4,
    'class_names': ["0", "1", "2", "3"],

    'val_interval': 1,  # validate after number of epochs
    'save_interval': 3,  # save model after number of epochs
    'epochs': 60
})

#=========================================================
config = base_config.copy()


