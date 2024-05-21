# FYRP
 First Year Research Project
filter_bank_csp.py Holds some functions needed to run the code, I should rename it, i thought the main approach i will look into is a filter bank.\n


The main notebooks in this Github are sliding_window.ipynb for the starting approach of just using windows, first I used smaller 0.2s windows without overlap to test wether or not such small windwos can make suitable predictions, currently it uses 2s windwos with a overlap of 1.8 seconds to still make predictions every 200ms. filter_bank.ipynb for the filter bank approach, using the smaller 200ms windwos as having mutliples of the data (for each filter bank) and longer windwos is not working on my machine. And lastly now i made indiv_frequency looking at the IAF of the three well working individuals.
The figures located in the folder figures are made with the function provided from Ivo but something went wrong.
