import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import RobustScaler

papka_data = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/Data/'
papka_tests = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/TestResults/'
# NBER recessions
from datetime import datetime
usrec = pd.read_excel(papka_data+'USREC.xls',index_col=0)
# Get the federal funds rate data
from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds
dta_fedfunds = pd.Series(fedfunds, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))

# Plot the data
dta_fedfunds.plot(title='Federal funds rate', figsize=(12,3))
# Fit the model
# (a switching mean is the default of the MarkovRegession model)
mod_fedfunds = sm.tsa.MarkovRegression(dta_fedfunds, k_regimes=2)
res_fedfunds = mod_fedfunds.fit()
print(res_fedfunds.summary())
res_fedfunds.smoothed_marginal_probabilities[1].plot(title='Probability of being in the high regime', figsize=(12,3));
print(res_fedfunds.expected_durations)

# Fit the model
mod_fedfunds2 = sm.tsa.MarkovRegression(dta_fedfunds.iloc[1:], k_regimes=2, exog=dta_fedfunds.iloc[:-1])
res_fedfunds2 = mod_fedfunds2.fit()
print(res_fedfunds2.summary())
res_fedfunds2.smoothed_marginal_probabilities[0].plot(title='Probability of being in the high regime', figsize=(12,3));

# Get the federal funds rate data
from statsmodels.tsa.regime_switching.tests.test_markov_regression import areturns
dta_areturns = pd.Series(areturns, index=pd.date_range('2004-05-04', '2014-5-03', freq='W'))

# Plot the data
dta_areturns.plot(title='Absolute returns, S&P500', figsize=(12,3))

# Get the federal funds rate data
from statsmodels.tsa.regime_switching.tests.test_markov_regression import areturns
dta_areturns = pd.Series(areturns, index=pd.date_range('2004-05-04', '2014-5-03', freq='W'))
# dta_areturns.plot()
# Fit the model
mod_areturns = sm.tsa.MarkovRegression(dta_areturns.iloc[1:], k_regimes=2, exog=dta_areturns.iloc[:-1], switching_variance=True)
res_areturns = mod_areturns.fit()
print(res_areturns.summary())
# shapiro(res_areturns.resid)
# shapiro(np.random.normal(0,1,100))



from statsmodels.tsa.regime_switching.tests.test_markov_regression import ogap, inf
dta_ogap = pd.Series(ogap, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))
dta_inf = pd.Series(inf, index=pd.date_range('1954-07-01', '2010-10-01', freq='QS'))
# dta_ogap.plot()
exog = pd.concat((dta_fedfunds.shift(), dta_ogap, dta_inf), axis=1).iloc[4:]
# Fit the 2-regime model
mod_fedfunds3 = sm.tsa.MarkovRegression(dta_fedfunds.iloc[4:], k_regimes=2, exog=exog, switching_variance=True,trend='n')
res_fedfunds3 = mod_fedfunds3.fit()
res_fedfunds3.summary()


#### KeyRate study ###
disser_path = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/Data/Clean/'
dataset = pd.read_excel(disser_path+'All_variables_clean_scenario.xlsx',index_col=0)
dataset = dataset[(dataset.index>'2004-12-01')]
# scenarios = pd.read_excel(disser_path+'Scenarios.xlsx',index_col=0, sheet_name='Neutral')
# dataset = pd.concat([dataset,scenarios]).drop_duplicates()
def get_pcs(stand_ds,cols=['Yld15', 'Yld10', 'Yld5', 'Yld3', 'Yld1']):
    yields_ds = stand_ds[cols].dropna()
    yields_ds_cov = yields_ds.cov()
    eigenvalues, eigenvectors = np.linalg.eig(yields_ds_cov)  # Put data into a DataFrame
    eigenvalues, eigenvectors = eigenvalues * 1, eigenvectors * -1
    # eigenvectors_inverted = np.linalg.inv(np.matrix(eigenvectors))
    principal_components = yields_ds.dot(eigenvectors)
    principal_components.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    # pc_orthogonality = principal_components.corr()
    stand_ds['PC1'] = principal_components['PC1']
    stand_ds['PC2'] = principal_components['PC2']
    stand_ds['PC3'] = principal_components['PC3']
    #return stand_ds[['PC1','PC2','PC3']]
def normalize_dataset(cut_ds,log_columns,mean_columns,orig_columns):
    log_data = np.log(cut_ds[log_columns])
    mean_diff = cut_ds[mean_columns]  # .dropna()
    mean_data = pd.DataFrame(RobustScaler().fit(mean_diff).transform(mean_diff), columns=mean_diff.columns,
                             index=mean_diff.index)
    stand_ds = pd.concat([log_data, mean_data], axis=1)
    stand_ds = pd.concat([stand_ds, cut_ds[orig_columns]], axis=1)
    stand_ds['Slope'] = stand_ds['Yld15'] - stand_ds['Yld1']
    stand_ds['Curve'] = (stand_ds['Yld15'] + stand_ds['Yld1'] - 2 * stand_ds['Yld5'])
    return stand_ds
log_columns = ['GDP_real2','PMImanfacturing', 'PMIservices','Balance_GDP','total_output_sa','DebtGDP','USDRUB','USDRUB_std','Urals', 'Monetization_coef','MOSPRIME_ROISFIX', 'Ruonia spot', 'Ruonia 1M', 'Ruonia F3M6M','KeyRate','ExpInfl_sa2', 'LIBOR_SOFR3M','OFZ_Nonrez_%','Funds']
mean_columns = ['Yld15', 'Yld10', 'Yld5', 'Yld3', 'Yld1','total_output_gap','Budget_balance_deseasoned','Total_infl','Exp_infl_growth']
orig_columns = []
stand_ds = normalize_dataset(dataset,log_columns,mean_columns,orig_columns)
get_pcs(stand_ds,cols=['Yld15', 'Yld10', 'Yld5', 'Yld3', 'Yld1'])
# stand_ds[['Slope','PC2']].dropna().plot()
stand_ds['PC1.L1'] = stand_ds['PC1'].shift()
stand_ds['Monetization_coef.D1'] = stand_ds['Monetization_coef'].pct_change()
stand_ds['Ruonia F3M6M.D1'] = stand_ds['Ruonia F3M6M'].pct_change()
stand_ds['Ruonia 1M.D1'] = stand_ds['Ruonia 1M'].pct_change()
stand_ds['Ruonia spot.D1'] = stand_ds['Ruonia spot'].pct_change()
stand_ds['ExpInfl_sa2.D1'] = stand_ds['ExpInfl_sa2'].pct_change()
stand_ds['Total_infl.D1'] = stand_ds['Total_infl'].pct_change()
stand_ds['Exp_infl_growth.D1'] = stand_ds['Exp_infl_growth'].pct_change()
stand_ds['KeyRate.D1'] = stand_ds['KeyRate'].pct_change()
stand_ds['USDRUB.D1'] = stand_ds['USDRUB'].pct_change()
ds = stand_ds[['PC1','PC1.L1','USDRUB.D1','Ruonia spot.D1','Exp_infl_growth.D1','KeyRate.D1','ExpInfl_sa2.D1','Monetization_coef.D1','Total_infl.D1']]#.dropna()
ds = ds.dropna()
# ds= ds.iloc[:-1]
# Fit the 2-regime model
# ds_train = ds[ds.index].dropna()
# ds_test = ds[ds.index>'2023-12-01']
mod_fedfunds3 = sm.tsa.MarkovRegression(ds['PC1'], k_regimes=2, exog=ds[['PC1.L1','Ruonia spot.D1','Monetization_coef.D1']], switching_variance=True,trend='c')
# mod_fedfunds3 = sm.tsa.MarkovAutoregression(ds_train['PC1'], k_regimes=2, order=1,exog=ds_train[['Ruonia spot.D1','Monetization_coef.D1']], switching_variance=True,trend='c')
res_fedfunds3 = mod_fedfunds3.fit()
res_fedfunds3.summary()
predicted_pc1 = res_fedfunds3.predict(start ='2023-12-01', end='2024-12-01')
predicted_pc1.to_excel(papka_tests+'MarkovModel_predict1.xlsx')
# pd.DataFrame({'Coefs':res_fedfunds3.tvalues,'Pvals':res_fedfunds3.pvalues}).to_excel(papka_tests+'MarkovModel.xlsx')

res_fedfunds3.resid.hist()
resid_norm = np.where(res_fedfunds3.resid>3,0,res_fedfunds3.resid)
plt.hist(resid_norm)
(np.mean(resid_norm**2))**(1/2)
from scipy.stats import shapiro
from scipy.stats import kstest
shapiro(resid_norm)
shapiro(np.random.normal(0,1,100))
kstest(np.random.normal(0,1,1000),cdf='norm')
kstest(resid_norm,cdf='norm')

ds['PC1 fitted'] = res_fedfunds3.fittedvalues
ds[['PC1','PC1 fitted','Ruonia F3M6M.D1']].plot()
res_fedfunds3.resid.plot()

res_fedfunds3.smoothed_marginal_probabilities[1].plot(title='Probability of being in the high regime', figsize=(12,3));
pd.DataFrame(res_fedfunds3.smoothed_marginal_probabilities[0], index=ds.index).plot()
# fig = sm.graphics.influence_plot(prestige_model, criterion="cooks")
# fig.tight_layout(pad=1.0)
