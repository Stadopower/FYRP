import numpy as np
import mne
from mne.preprocessing import read_ica
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP, SPoC
from continuous_control_bci.util import emg_classes_to_eeg_classes, SUBJECT_IDS
from continuous_control_bci.data.load_data import load_calibration, load_driving
from continuous_control_bci.data.preprocessing import make_epochs, epochs_to_train_test
from continuous_control_bci.modelling.csp_classifier import create_csp_classifier, get_driving_epochs_for_csp
from scipy.signal import cheby2, sosfiltfilt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def filter_bank(raw:np.ndarray, freq_bands, order:int):
    filtered_signals = [[] for _ in range(len(freq_bands))]
    for i, freq_band in enumerate(freq_bands):
        iir_params = dict(order=order, ftype="cheby2", output='sos', rs=10) # what exactly is rs? How do i know which order to use?
        iir_params = mne.filter.construct_iir_filter(iir_params, f_pass=freq_band, sfreq=2048, btype="bandpass")
        filtered_raw = sosfiltfilt(iir_params['sos'], raw)
        filtered_signals[i].append(filtered_raw)
    filtered_signals = np.asarray(filtered_signals)
    filtered_signals = filtered_signals[:,0]
    return filtered_signals


def filter_bank_csp(raw:np.ndarray, y_train, freq_bands, order:int, rank):
    csp = CSP(n_components=6, reg='shrinkage', log=True, rank=rank, transform_into='average_power')
    csp_channels = []
    csp_info = []
    for i, freq_band in enumerate(freq_bands):
        iir_params = dict(order=order, ftype="cheby2", output='sos', rs=10) # what exactly is rs? How do i know which order to use?
        iir_params = mne.filter.construct_iir_filter(iir_params, f_pass=freq_band, sfreq=2048, btype="bandpass")
        filtered_raw = sosfiltfilt(iir_params['sos'], raw)
        csp.fit(filtered_raw, y_train)
        tmp = csp.transform(filtered_raw)
        csp_info.append(csp)
        if i == 0:
            csp_channels = tmp
        else:
            csp_channels = np.concatenate((csp_channels, tmp), axis=1)
    return csp_channels, csp_info


def predict_against_threshold(clf, X, t):
    y_driving_pred_prob = clf.predict_proba(X)
    rest = y_driving_pred_prob[:, 2] > t
    y_driving_pred = np.array([2] * len(rest))
    y_driving_pred[~rest] = y_driving_pred_prob[~rest, :2].argmax(axis=1)
    return y_driving_pred


def predict_against_threshold_indiv(clf, X, t):
    y_driving_pred_prob = clf.predict_proba(X)
    if y_driving_pred_prob [:, 2] > t:
        y_pred = 2
    elif y_driving_pred_prob [:, 0] > y_driving_pred_prob [:, 1]:
        y_pred = 0
    else:
        y_pred = 1
    return y_pred


def create_csp_svm_classifier(X_train: np.ndarray, y_train: np.ndarray, rank):
    """
    Trains a CSP classifier on all the data.
    First, however it runs 5-fold cross validation to make cross-validated predictions.
    This the resulting predictions are returns for a fair evaluation, with an optimal model for the training data.
    :param X_train:
    :param y_train:
    :return:
    """
    clf_eeg = Pipeline([
        ("CSP", CSP(n_components=6, reg='shrinkage', log=True, rank=rank)),
        ("SVC", SVC(C=10, gamma=0.1, kernel="rbf"))]) #'SVC__C': 10, 'SVC__gamma': 0.1, 'SVC__kernel': 'rbf'
    #param_grid = {'SVC__C': [0.1, 1, 10, 100, 1000],
    #              'SVC__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #              'SVC__kernel': ['rbf', 'gauss']}
    #    search = GridSearchCV(clf_eeg, param_grid, n_jobs=2, cv=5)
    #    search.fit(X_train, y_train)
    #    print("BEST PARAMETERS!!!!!!!",search.best_params_)
    #    params = search.get_params()

    y_pred = cross_val_predict(clf_eeg, X_train, y_train, cv=5)

    clf_eeg.fit(X_train, y_train)

    return clf_eeg, y_pred #params


def majority_pred(X, clf, n):
    """

    :param X: Data to make predictions on
    :param clf: Classifier to use
    :param n: number of epochs that should be aggregated over
    :return: Returns both the individual predictions for each window,
     as well as the predictions for an aggregated 2 second window
    """
    y_preds = []
    majority_pred = []
    # left, right, rest
    pred_direct = [0,0,0]
    for i, epoch in enumerate(X):
        # Predicting each 200ms windows individually
        y_driving_pred = predict_against_threshold_indiv(clf, X[i:i+1,:,:], 0.2)
        pred_direct[y_driving_pred] += 1
        y_preds.append(y_driving_pred)

        # Now aggregate over 2sec time window
        if i % n == 0:
            majority_pred.append(pred_direct.index(max(pred_direct)))
            pred_direct = [0,0,0]


    y_preds = np.asarray(y_preds)
    majority_pred = np.asarray(majority_pred)
    return y_preds, majority_pred