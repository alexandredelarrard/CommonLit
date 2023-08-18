from steps.step_data_load import DataLoader
from steps.step_data_prep import DataPreparation
from data_models.modelling_lightgbm import TrainModel

import numpy as np

if __name__ == "__main__":

    config_path = "./configs/main.yml"
    submit= False

    # data loading
    loader = DataLoader(config_path)
    datas= loader.run()

    # # data prep
    # prep = DataPreparation()
    # train, test = prep.run(datas)

    # # training 
    # train_step = TrainModel(loader.config.modelling.config_lgbm, data=train)