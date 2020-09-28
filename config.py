from utils import normabspath
import os

dirname: str = os.path.dirname(__file__)

datapaths = {'tsa': {'train_1': normabspath(dirname, 'tsa/data/train_1.csv')}}
