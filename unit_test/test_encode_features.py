import warnings
warnings.filterwarnings("ignore")
from utils import *
from constants import *
import pandas as pd


class TestEncodeFeatures:
    def test_one_hot_len(self):
       inference_data, labels = get_inference_data()
       one_hot_encode_features=encode_features(inference_data, inference_data)
       columns_encode_features = one_hot_encode_features.columns
       assert len(columns_encode_features) == len(ONE_HOT_ENCODED_FEATURES), "One Hot Encoding does not match"








