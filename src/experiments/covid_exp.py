import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.models import rnn, lstm, bjrnn, cfrnn, copulaCPTS, dplstm, vanila_copula

import logging
import datetime


class COVIDDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, sequence_lengths):
        super(COVIDDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.sequence_lengths[idx]

def configure_logging():
    # Generate a unique filename for the log file based on the current timestamp
    log_filename = f"./debugging/app_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    # Configure logging to write to the unique log file
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)


def experiment(train, valid, test, target_len=50, name="exp"):
    configure_logging()

    rnn_model = rnn.rnn(
        input_size=1, embedding_size=128, output_size=1, horizon=target_len
    )
    encdec_model = lstm.lstm_seq2seq(
        input_size=1, output_size=1, embedding_size=128, target_len=target_len
    )
    models = [encdec_model, rnn_model]#[][rnn_model]#, encdec_model]

    x_train = train.X
    y_train = train.Y

    logging.debug('begin train')
    for m in models:
        print(m)
        m.train_model(x_train, y_train, n_epochs=200, batch_size=64)

        print('m trained')
        logging.debug('finished training m')

    print('start saved models')
    logging.debug('save')
    with open("./trained/200models_%s.pkl" % name, "wb") as f:
        pickle.dump(models, f)
    logging.debug('saved models')

    x_cali = valid.X
    y_cali = valid.Y

    UQ = {}
    logging.debug('start dprnn')
    dprnn = dplstm.DPRNN(
        epochs=150, input_size=1, output_size=1, n_steps=target_len, dropout_prob=0.2
    )
    dprnn.fit(x_train, y_train)
    UQ["dprnn"] = dprnn


    #logging.debug('start bj')
    #bj_class = bjrnn.bj_rnn(models[1], recursion_depth=15)
    #logging.debug('initialized bj')
    #bj_class.LOBO_residuals = np.expand_dims(bj_class.LOBO_residuals, axis=-1)
    #logging.debug('bj residuals')
    #UQ["bjrnn"] = bj_class

    logging.debug('start cf')
    cf = cfrnn.CFRNN(models[1], x_cali, y_cali)
    logging.debug('cf calbirate')
    cf.calibrate()
    UQ["cfrnn"] = cf

    cf = cfrnn.CFRNN(models[0], x_cali, y_cali)
    cf.calibrate()
    UQ["cf-EncDec"] = cf

    logging.debug('copula')
    copula = copulaCPTS.copulaCPTS(models[1], x_cali, y_cali)
    copula.calibrate()
    UQ["copula-rnn"] = copula

    copula = copulaCPTS.copulaCPTS(models[0], x_cali, y_cali)
    copula.calibrate()
    UQ["copula-EncDec"] = copula

    vanilla = vanila_copula.vanila_copula(models[1], x_cali, y_cali)
    vanilla.calibrate()
    UQ["vanila-copula"] = vanilla
    
    logging.debug('trained vanilla')

    x_test = test.X
    y_test = test.Y 

   
    areas = {}
    coverages = {}

    epsilon_ls = np.linspace(0.05, 0.50, 10)

    logging.debug('uq')

    for k, uqmethod in UQ.items():
        print(k)
        area = []
        coverage = []
        for eps in epsilon_ls:
            pred, box = uqmethod.predict(x_test, epsilon=eps)
            area.append(uqmethod.calc_area_1d(box))
            pred = torch.tensor(pred)
            coverage.append(uqmethod.calc_coverage_1d(box, pred, y_test))
        areas[k] = area
        coverages[k] = coverage

    with open("./trained/uq_%s.pkl" % name, "wb") as f:
        pickle.dump(UQ, f)
    with open("./trained/results_%s.pkl" % name, "wb") as f:
        pickle.dump((areas, coverages), f)

    return areas, coverages, (models, UQ)


def main():
    target_len = 50
    with open("processed_data/covid_conformal_%d.pkl" % target_len, "rb") as f:
        train_dataset, calibration_dataset, test_dataset = pickle.load(f)

    for i in range(3):
        res = experiment(
            train_dataset,
            calibration_dataset,
            test_dataset,
            target_len=target_len,
            name="covid_daily_vanilla" + str(i),
        )  # target len 6
        print("finished run " + str(i))

        del res


if __name__ == "__main__":
    main()
