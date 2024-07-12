# FYRP
 First Year Research Project 

filter_bank_csp.py Holds some functions needed to run the code, not only for the filter bank approach but also other functions.
Not all of them were ultimately used, some were just used for experimenting with different things,
like using majority predictons for larger windows.


There is one main notebook for each approach. sliding_windows.ipynb for creating the baseline predictions for the three and two class problem. filter_bank.ipynb for the filter bank approach and indiv_frequency.ipynb for the individual frequency approach. 
Inside the filter_bank notebook some investigation on training and testing on calibration and training and testing on the
driving data alone was conducted. Inside both the sliding approach and the individual_frequency notebook the reversing training and testing was investigated.

Some listing of results can be found in the results.ipynb (Some are named _maj because I tried around using a majority prediction for larger windows but then stopped doing it, and to be extra sure some are names _not_maj because I once started running the code and saved the files wrong) and some investigation of ERDS, PSD and CSP was done in pattern_analysis.ipynb.

Overall the different approaches to increase the performance did not work and one of the main problems that was found was the difference in frequency peaks between calibration and testing data.

In the data folder, the results of the run participants for the different approaches can be found stored in pickle files,
while the figures are stored in the figures folder.