import pandas as pd
import os


def normabspath(basedir: str, filename: str):
    return os.path.normpath(os.path.join(basedir, filename))


dirname: str = os.path.dirname(__file__)

datapaths = {'banknotes': normabspath(dirname, 'data/banknotes.csv'),
             'darts': normabspath(dirname, 'data/darts.csv'),
             'irrigation_machine': normabspath(dirname, 'data/irrigation_machine.csv')}

