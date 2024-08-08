import pandas as pd
import numpy as np

class LoadData:
    @staticmethod
    def read_data(file_path):
        data = pd.read_excel(file_path)
        return data.values

class DataPreprocessing:
    @staticmethod
    def min_max_norm(data):
        minVal = np.min(data, axis=0)
        maxVal = np.max(data, axis=0)
        normalizedData = (data - minVal) / (maxVal - minVal)
        return normalizedData
