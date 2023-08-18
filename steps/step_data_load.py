#
# -*- coding: utf-8 -*-

import pandas as pd
from steps.step import Step

class DataLoader(Step):

    def __init__(
        self,
        config_path: str
    ):
        super().__init__(config_path=config_path)
        self.base_path = self.config.load.base["data_path"]

    def run(self):
        
        # load all datas
        datas = self.data_loading()

        return datas
    
    def data_loading(self):

        datas = {}

        datas["prompt_train"] = pd.read_csv(self.base_path + "/prompts_train.csv")
        datas["prompt_test"] = pd.read_csv(self.base_path + "/prompts_test.csv")
        datas["summaries_train"] = pd.read_csv(self.base_path + "/summaries_train.csv")
        datas["summaries_test"] = pd.read_csv(self.base_path + "/summaries_test.csv")
        datas["sample_submission"] = pd.read_csv(self.base_path + "/sample_submission.csv")
    
        return datas

        