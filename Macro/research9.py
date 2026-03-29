import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar import vecm
from statsmodels.tsa import ardl
from scipy import stats
import ruptures as rpt
import matplotlib.cm as cm
from hmmlearn import hmm
from scipy.stats import kstest, shapiro
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

cmap = cm.get_cmap('tab10')

def perform_adf_test(series):
    result = adfuller(series,regression='c')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return result
def perform_kpss_test(series):
    """
    https://www.youtube.com/watch?v=ubzH1BJuUro&t=5s
    H0: The time series is trend stationary.
    HA: The time series is not trend stationary.
    If the of the test is less than some significance level (e.g. α = .05) then we reject the null hypothesis and conclude that the time series is not trend stationary.
    """
    result = kpss(series,regression='c')
    print('KPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return result
def perform_stationarity_tests(dataset, variables):
    adfres_total = []
    kpssres_total = []
    for col in variables:
        # col = 'ExpInfl_sa'
        adfres_total.append(perform_adf_test(dataset[col].dropna())[1])
        kpssres_total.append(perform_kpss_test(dataset[col].dropna())[1])

    stat_res = pd.DataFrame({'ADF':adfres_total,'KPSS':kpssres_total}, index=variables).T
    return stat_res
    # pd.concat([kpssres_df, adfres_df]).to_excel(article4_path+'adfkpss_regressors.xlsx')
def coint_tests(dataset, list1, list2, format):
    # list1 = ['Yld15', 'Yld10', 'Yld5', 'Yld3', 'Yld1']
    # list2 = ['SAAR','KeyRate','ExpInfl_sa']
    coint_stats_r0 = pd.DataFrame(columns=list1, index=list2)
    coint_stats_r1 = pd.DataFrame(columns=list1, index=list2)
    coint_stats_r3 = pd.DataFrame(columns=list1, index=list2)
    coint_stats_r4 = pd.DataFrame(columns=list1, index=list2)
    eg_stats = pd.DataFrame(columns=list1, index=list2)
    for l1 in list1:
        for l2 in list2:
            # l1='Yld10'
            # l2='Inflation'
            # dataset_ardl = dataset_ols_cut[[l1,l2]]
            dataset_ardl = dataset[[l1, l2]].dropna()
            lag_order = vecm.select_order(data=dataset_ardl, maxlags=10, deterministic="ci")
            """ -1 - no deterministic terms 0 - constant term 1 - linear trend"""
            cointres_linear = vecm.coint_johansen(dataset_ardl, det_order=format, k_ar_diff=lag_order.aic)
            rank_trace = vecm.select_coint_rank(dataset_ardl, det_order=format, k_ar_diff=lag_order.aic,
                                                method="trace", signif=0.05)
            rank_eig = vecm.select_coint_rank(dataset_ardl, det_order=0, k_ar_diff=1,
                                              method="maxeig", signif=0.05)
            print(l1,l2)
            print(rank_trace.summary())
            print(rank_eig.summary())
            # cointres_linear.trace_stat
            eg = sm.tsa.stattools.coint(dataset_ardl[l1], dataset_ardl[l2], trend='ct')
            coint_stats_r0.at[l2,l1] = str(round(cointres_linear.trace_stat[0],2))+'/'+str(round(cointres_linear.max_eig_stat[0],2))
            coint_stats_r1.at[l2,l1] = str(round(cointres_linear.trace_stat[1],2))+'/'+str(round(cointres_linear.max_eig_stat[1],2))
            coint_stats_r3.at[l2, l1] = round(cointres_linear.trace_stat[0], 2)
            coint_stats_r4.at[l2, l1] = round(cointres_linear.trace_stat[1], 2)
            eg_stats.at[l2,l1] = str(round(eg[0],2))+'/('+str(round(eg[1],2))+')'
    return coint_stats_r0, coint_stats_r1, cointres_linear.trace_stat_crit_vals, cointres_linear.max_eig_stat_crit_vals, eg_stats, coint_stats_r3,coint_stats_r4
def PACF_ACF_plots(series):
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(series, lags=10, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(series, lags=10, ax=ax2)
    plt.show()
def getlogs(ds,cols):
    for i in cols:
        try:
            ds[i+'_log'] = np.log(ds[i])
        except TypeError:
            pass
    return ds
def getdiffs(ds):
    for i in ds.columns:
        try:
            ds[i+'.D1'] = ds[i].diff()
        except TypeError:
            pass
    return ds
def check_none_values(obs, data):
    for key in obs:
        if key in data and data[key] is None:
            return True
    return False
def ardl_data(cut_ds2, y_name, regs, balance_none, morder=2):
    lag_order_ardl = ardl.ardl_select_order(endog=cut_ds2[y_name], maxlag=1, exog=cut_ds2[regs], maxorder=morder, trend='n')
    aicvals = lag_order_ardl.aic[lag_order_ardl.aic.index[:1000]]
    for index, value in aicvals.items():
        # print(index)
        # if (value[1][balance_none[0]] is None) or (value[1][balance_none[1]] is None):
        if check_none_values(balance_none, value[1]):
            aicvals.drop(index, inplace=True)
    x = aicvals.loc[aicvals.index[0]]
    dl_lags = {key: value for key, value in x[1].items() if value is not None}
    print(lag_order_ardl.ar_lags, dl_lags)
    ardlmodel = ardl.ARDL(endog=cut_ds2[y_name], lags=lag_order_ardl.ar_lags, exog=cut_ds2[regs], order=dl_lags,trend='n').fit()
    return ardlmodel
def get_pcs(ds,cols=['Yld1','Yld3','Yld5', 'Yld10', 'Yld15']):
    yields_ds = ds[cols].dropna()
    yields_ds_cov = yields_ds.cov()
    eigenvalues, eigenvectors = np.linalg.eig(yields_ds_cov)  # Put data into a DataFrame
    eigenvalues, eigenvectors = eigenvalues * 1, eigenvectors * -1
    # eigenvectors_inverted = np.linalg.inv(np.matrix(eigenvectors))
    principal_components = yields_ds.dot(eigenvectors)
    principal_components.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    # pc_orthogonality = principal_components.corr()
    ds['PC1'] = principal_components['PC1']
    ds['PC2'] = principal_components['PC2']
    ds['PC3'] = principal_components['PC3']
    return ds[['PC1','PC2','PC3']], eigenvectors, eigenvalues
def normalize_dataset(cut_ds,log_columns,mean_columns,orig_columns):
    log_data = np.log(cut_ds[log_columns])
    mean_diff = cut_ds[mean_columns]  # .dropna()
    scaler = RobustScaler()
    mean_data = pd.DataFrame(RobustScaler().fit(mean_diff).transform(mean_diff), columns=mean_diff.columns,
                             index=mean_diff.index)
    stand_ds = pd.concat([log_data, mean_data], axis=1)
    stand_ds = pd.concat([stand_ds, cut_ds[orig_columns]], axis=1)
    return stand_ds, scaler
def normalize_yields(cut_ds,mean_columns):
    mean_diff = cut_ds[mean_columns]  # .dropna()
    scaler = RobustScaler()
    mean_data = pd.DataFrame(RobustScaler().fit(mean_diff).transform(mean_diff), columns=mean_diff.columns,
                             index=mean_diff.index)
    mean_data.columns = ['Yld1','Yld3','Yld5', 'Yld10', 'Yld15']
    return mean_data, scaler
def build_regplots(ds, stat_vars, target):
    xxc = [x for x in stat_vars if x != target]
    g = sns.FacetGrid(pd.DataFrame(xxc), col=0, col_wrap=6, sharex=False, )
    for ax, x_var in zip(g.axes, xxc):
        # sns.scatterplot(cut_ds2d_clean, x=x_var, y='Yld10', ax=ax)
        sns.regplot(ds, x=x_var, y=target, ax=ax)
        ax.set(xlabel=None)
def plot_comparison(stand_ds,name1,name2):
    ds = stand_ds[[name1,name2]].dropna()
    plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(ds.index, ds[name1])
    ax1.legend([name1], loc="upper right")

    ax2 = plt.subplot(212, sharex = ax1)
    ax2.plot(ds.index, ds[name2])
    ax2.legend([name2],loc="upper right")

#plots & data
papka_plots = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/Plots/'
papka_data = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/Data/'
papka_clean = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/Data/Clean/'
papka_zerocurve = '/Users/mmajidov/Desktop/АСПА/Disser/Статья 3/Data/Gcurve_daily.xlsx'
disser_path = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/Data/Clean/'
article3_path = '/Users/mmajidov/Desktop/АСПА/Disser/Статья 3/Data/'
article4_path = '/Users/mmajidov/Desktop/АСПА/Disser/Статья 4/Data/'
papka_tests = '/Users/mmajidov/Desktop/АСПА/Disser/Диссер/TestResults/'
common_path = '/Users/mmajidov/Desktop/АСПА/Disser/Статья 3/Data/'

dataset = pd.read_excel(disser_path+'All_variables_clean_scenario.xlsx',index_col=0)
cut_ds = dataset[(dataset.index>'2004-12-01')]
unemployment_breakdown = seasonal_decompose(cut_ds['Норма безработицы (на начало месяца), %'].dropna(), model="additive", two_sided=False)
cut_ds['Unemployment_sa'] = unemployment_breakdown.trend

cut_ds1 = cut_ds[(cut_ds.index>'2004-12-01') & (cut_ds.index<'2024-12-01')]#.drop(columns=['OFZ_Nonrez_%','ExpInfl_sa2'])
del cut_ds, dataset
loggedcols = ['GDP_real2','PMIservices','Monetization_coef','MOEX','Urals','KeyRate','ExpInfl_sa2','DebtGDP','total_output_sa','Unemployment_sa','Реальная зарплата с поправкой на сезонность янв93(факт) =100, на начало месяца']
cols = [x for x in cut_ds1.columns if 'Yld' not in x]
# cols.remove('S&P'),cols.remove("Moody's"),cols.remove('Fitch')
for x in ['S&P',"Moody's",'Fitch','Dummy','Period']:
    cols.remove(x)
cut_ds1 = getlogs(cut_ds1, loggedcols)
cut_ds1 = getdiffs(cut_ds1)

# scaler1 = RobustScaler()
# meands = pd.DataFrame(scaler1.fit(cut_ds1[cols]).transform(cut_ds1[cols]), columns=cols,index=cut_ds1.index)
# meands1 = getlogs(meands, loggedcols)
# meands1 = getdiffs(meands)
diff_columns = ['GDP_real2.D1','PMIservices_log.D1','Balance_GDP.D1','total_output_sa_log.D1','USDRUB.D1','Urals.D1', 'Monetization_coef.D1','Ruonia spot.D1','KeyRate_log.D1','Budget_balance_deseasoned.D1','Total_infl.D1']
# cut_ds1[diff_columns].hist(bins=50, figsize=(15, 10))
mean_diff = cut_ds1[['Yld1','Yld3','Yld5', 'Yld10', 'Yld15']]  # .dropna()
scaler = RobustScaler()
mean_data = pd.DataFrame(scaler.fit(mean_diff).transform(mean_diff), columns=mean_diff.columns,index=mean_diff.index)
pcs, eigenvectors,eigenvalues = get_pcs(mean_data,cols=['Yld1','Yld3','Yld5','Yld10','Yld15'])

pcs['PC2'] =pcs['PC2']*-1
total_ds = pd.concat([pcs,cut_ds1],axis=1) #cut_ds1
total_ds[['Yld1','Yld3','Yld5', 'Yld10', 'Yld15']] = mean_data[['Yld1','Yld3','Yld5', 'Yld10', 'Yld15']]
total_ds['Level'] = (total_ds['Yld1']+total_ds['Yld3']+total_ds['Yld5']+total_ds['Yld10']+total_ds['Yld15'])/5
total_ds['Slope'] = (total_ds['Yld15']-total_ds['Yld1'])
total_ds['Curvature'] = (total_ds['Yld15']+total_ds['Yld1']-2*total_ds['Yld5'])
total_ds['Urals.D1'] = total_ds['Urals'].diff()
total_ds['PC1.D1'] = total_ds['PC1'].diff()
total_ds['PC2.D1'] = total_ds['PC2'].diff()
total_ds['PC3.D1'] = total_ds['PC3'].diff()
total_ds['PC1.L1'] = total_ds['PC1'].shift()
total_ds['PC2.L1'] = total_ds['PC2'].shift()
total_ds['PC3.L1'] = total_ds['PC3'].shift()
total_ds['Balance_GDP_log'] = np.log(total_ds['Balance_GDP'])
total_ds['Balance_GDP_log.D1'] = total_ds['Balance_GDP_log'] - total_ds['Balance_GDP_log'].shift(1)
total_ds['Urals_log'] = np.log(total_ds['Urals'])
total_ds['Urals_log.D1'] = total_ds['Urals_log'] - total_ds['Urals_log'].shift(1)
total_ds['Unemployment_sa.D1'] = total_ds['Unemployment_sa'].diff()
total_ds['Monetary_act'] = (total_ds['Ruonia spot']/100) * total_ds['Total_infl']
total_ds['Monetary_unemp'] = (total_ds['Ruonia spot']/100) * total_ds['Unemployment_sa']/100
total_ds['Shortspread'] = total_ds['Ruonia spot'] - total_ds['Ruonia F3M6M']
# total_ds['Shortspread'].plot()

total_ds['Monetary_act.D1'] = total_ds['Monetary_act'].diff()
total_ds['Monetary_unemp.D1'] = total_ds['Monetary_unemp'].diff()

# plots
# ds22 = total_ds[total_ds.index>'2022-03-01']
# fig, ax = plt.subplots(3, sharex='col', sharey='row',figsize=(8, 10))
# ax[0].plot(total_ds.index, total_ds[['PC1','Level']])
# ax[1].plot(total_ds.index, total_ds[['PC2','Slope']])
# ax[2].plot(total_ds.index, total_ds[['PC3','Curvature']])
# fig.savefig(papka_plots+"decomposition_full.svg", format="svg")

# fig, ax = plt.subplots(1, 3, sharex='col', sharey=False,figsize=(8, 4))
# ax[0].plot(total_ds.index, total_ds[['PC1','Level']])
# ax[0].legend(['PC1','Level'],loc='upper right')
# ax[1].plot(total_ds.index, total_ds[['PC2','Slope']])
# ax[1].legend(['PC2','Slope'],loc='upper right')
# ax[2].plot(total_ds.index, total_ds[['PC3','Curvature']])
# ax[2].legend(['PC3','Curvature'],loc='upper right')
# fig.savefig(papka_plots+"decomposition_full_horizontal.svg", format="svg")




# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('time')
# ax1.set_ylabel('Urals', color=color)
# ax1.plot(total_ds.index, total_ds['Urals'], color=color)
# ax1.tick_params(axis='y', labelcolor=color)
# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('PC2', color=color)  # we already handled the x-label with ax1
# ax2.plot(total_ds.index, total_ds['PC2'], color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# tds[['Urals','PC2','Slope']].corr()
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
# fig.savefig(papka_plots+"Urals_PC2.svg", format="svg")

# total_ds[['PC2','Slope']].plot()
# total_ds[total_ds.index<='2022-01-01'][['ExpInfl_sa2','PC2']].dropna().plot()

# PC1 #
ds = total_ds[['PC1','PC1.L1','USDRUB.D1','KeyRate.D1','Ruonia spot','Monetization_coef.D1','Total_infl','Total_infl.D1','total_output_gap']]#.dropna()
ds = ds.dropna()
mod_fedfunds3 = sm.tsa.MarkovRegression(ds['PC1'], k_regimes=2, exog=ds[['PC1.L1','Ruonia spot','Total_infl','total_output_gap']], switching_variance=True,trend='c')
# mod_fedfunds3 = sm.tsa.MarkovAutoregression(ds_train['PC1'], k_regimes=2, order=1,exog=ds_train[['Ruonia spot.D1','Monetization_coef.D1']], switching_variance=True,trend='c')
res_fedfunds3 = mod_fedfunds3.fit()
res_fedfunds3.summary()
model1 = res_fedfunds3.summary().tables[1].as_html()
model2 = res_fedfunds3.summary().tables[2].as_html()
probs = res_fedfunds3.summary().tables[3].as_html()
model1_pds = pd.read_html(model1, header=0, index_col=0)[0]
model2_pds = pd.read_html(model2, header=0, index_col=0)[0]
probs_pds = pd.read_html(probs, header=0, index_col=0)[0]
model1_pds.to_excel(papka_tests+'taylor_r1.xlsx')
model2_pds.to_excel(papka_tests+'taylor_r2.xlsx')
probs_pds.to_excel(papka_tests+'probs.xlsx')


fig, axes = plt.subplots(2, figsize=(10, 7))

ax = axes[0]
ax.plot(res_fedfunds3.smoothed_marginal_probabilities[0])
ax.set(title="Smoothed probability of a low-variance regime for first component")

ax = axes[1]
ax.plot(res_fedfunds3.smoothed_marginal_probabilities[1])
ax.set(title="Smoothed probability of a high-variance regime for first component")
fig.tight_layout()
plt.savefig(papka_plots + "Transition probabilities.svg", format="svg")
from sklearn.metrics import mean_squared_error, r2_score
res_fedfunds3.fittedvalues
mean_squared_error(res_fedfunds3.data.endog,res_fedfunds3.fittedvalues)
r2 = r2_score(res_fedfunds3.data.endog,res_fedfunds3.fittedvalues)
drob = ((len(res_fedfunds3.data.endog)-1)/(len(res_fedfunds3.data.endog)-res_fedfunds3.data.exog.shape[1]-1))
r2_adj = 1-(1-r2)*drob
r2_adj

res_fedfunds3.f_test()
compds = pd.concat([res_fedfunds3.fittedvalues,ds['PC1']],axis=1)
compds.columns = ['Fitted','PC1']
compds.plot()
ds.index = pd.to_datetime(ds.index, infer_datetime_format=True)
predicted_pc1 = res_fedfunds3.predict(start='2024-09-01', end='2024-12-01')
res_fedfunds3.smoothed_marginal_probabilities[1].plot(title='Probability of being in the high regime', figsize=(12,3));
pd.DataFrame(res_fedfunds3.smoothed_marginal_probabilities[0], index=ds.index).plot()

# pd.DataFrame({'Coefs':res_fedfunds3.tvalues,'Pvals':res_fedfunds3.pvalues}).to_excel(papka_tests+'MarkovModel.xlsx')
#### tests for structural break ####
ds = total_ds[['PC2','PC3']].dropna()

ds_before22 = ds[ds.index<'2022-03-01']
ds_after22 = ds[ds.index>='2022-03-01']


model1 = sm.tsa.AutoReg(ds_before22['PC3'], lags=1, trend='n').fit()
model2 = sm.tsa.AutoReg(ds_after22['PC3'], lags=1, trend='n').fit()
# Extract AR coefficients
ar_coefs1 = model1.params
ar_coefs2 = model2.params
model1.summary()
model2.summary()

from scipy.stats import chi2

# Get the covariance matrix for both models
cov_matrix1 = model1.cov_params()
cov_matrix2 = model2.cov_params()
# Compute the Wald statistic
diff = ar_coefs1 - ar_coefs2
cov_diff = cov_matrix1 + cov_matrix2  # assuming independent series
wald_stat = np.dot(np.dot(diff.T, np.linalg.inv(cov_diff)), diff)
# Degrees of freedom = number of coefficients (order of AR model)
p_value = chi2.sf(wald_stat, df=len(ar_coefs1))
print(f"Wald test statistic: {wald_stat}, p-value: {p_value}")

from scipy.stats import f

# Extract residual variances from both models
residuals1 = model1.resid
residuals2 = model2.resid
var1 = np.var(residuals1, ddof=1)
var2 = np.var(residuals2, ddof=1)
# Perform F-test for equal variances
F_stat = var1 / var2
dfn = len(residuals1) - 1  # degrees of freedom for sample 1
dfd = len(residuals2) - 1  # degrees of freedom for sample 2
p_value_f = f.cdf(F_stat, dfn, dfd)
print(f"F-test statistic: {F_stat}, p-value: {p_value_f}")


####### 2004-14 #####
cols = [x for x in total_ds.columns if 'Yld' not in x]
for x in ['S&P',"Moody's",'Fitch','Dummy','Period']:
    cols.remove(x)
pc_corrmat_source = total_ds[(total_ds.index>'2004-12-01') & (total_ds.index<'2022-01-01')][cols]
pc_corrmat = pc_corrmat_source.corr()[['PC3','PC3.D1']]
del pc_corrmat_source
pc_corrmat.sort_values(by='PC2',inplace=True)
pc_corrmat.loc['Urals']
# pc_corrmat.loc['Monetary_unemp.D1']
cut_ds1 = total_ds[(total_ds.index>'2004-12-01') & (total_ds.index<'2022-01-01')]
# PC2
'DebtGDP','total_output_sa'

fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = total_ds['KeyRate']
ydata =  total_ds['PC3'] #Норма безработицы (на начало месяца), %
zdata =  total_ds['total_output_gap']
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis');
'PC2.L1 Urals.D1 GDP_real.D1'
# sns.regplot(cut_ds1, x='PC2',y='OFZ_Nonrez_%')
# sns.regplot(cut_ds1, x='PC2',y='Денежный_рынок')
high_unemp = cut_ds1[cut_ds1['Норма безработицы (на начало месяца), %']<=6.5]
low_unemp = cut_ds1[cut_ds1['Норма безработицы (на начало месяца), %']>6.5]
sns.regplot(low_unemp, x='OFZ_',y='PC2')
# PACF_ACF_plots(cut_ds1['PC3'].dropna())
# ardl_model.fittedvalues.plot()
# ds1['PC2'].plot()
cds1 = total_ds[['PC2','PC2.L1','PC2.D1','ExpInfl_sa_diff.D1','ОФЗ_ОБР_РЕПО.D1','Monetization_coef.D1','total_output_gap','GDP_real2','Monetary_act','Monetary_unemp','Urals','Urals.D1','GDP.D1','Unemployment_sa.D1','ExpInfl_sa2.D1']].dropna()
perform_stationarity_tests(cds1, ['PC2','Monetary_act','Monetary_unemp'])
cds1['Monetary_unemp'].plot()
from statsmodels.tsa.stattools import grangercausalitytests
gc_res = grangercausalitytests(cds1[['PC2','Monetary_act']], 6)
gc_res = grangercausalitytests(cds1[['PC2','Urals']], 6)
gc_res = grangercausalitytests(cds1[['PC2','GDP_real2']], 6)
c1, c2, crit_trace, crit_eig, eg, c3, c4 = coint_tests(cut_ds1,['Urals'],['PC2'],0)
cds1['Monetary_act.L6'] =cds1['Monetary_act'].shift(6)
cds1=cds1.dropna()
real_reg_lvl = sm.OLS(cds1['PC2'], cds1[['PC2.L1','Monetary_act.L6']]).fit() #,'RateParity' #cov_type='HAC'
real_reg_lvl.summary()

regs_new = ['Urals','Urals_log','Urals_log.D1','total_output_gap','PMIservices','GDP.D1']
ds1 = total_ds[['PC2','PC2.L1','PC2.D1']+regs_new].dropna()
ds1['total_output_gap.D1'] =ds1['total_output_gap'].diff()
ds1['Urals2'] =ds1['Urals_log']**2
ds1_before22 = ds1[ds1.index<'2022-02-01']
model = sm.OLS(ds1_before22['PC2'],ds1_before22[['PC2.L1','Urals_log','Urals2']])
results = model.fit()  # You can adjust lags
results.summary()
# pd.concat([results.fittedvalues,ds1['PC2']],axis=1).plot()
ds1_before22['ECT'] = results.resid.shift(1)
ds1_before22=ds1_before22.dropna()
model = sm.OLS(ds1_before22['PC2.D1'],ds1_before22[['ECT']])
results = model.fit()  # You can adjust lags
results.summary()


model = sm.OLS(ds1['PC2'],ds1[['PC2.L1']])
results = model.fit()  # You can adjust lags
results.summary()
pd.concat([results.fittedvalues,ds1['PC2']],axis=1).plot()


regs_new = ['Urals','Ruonia spot','total_output_gap','PMIservices','ExpInfl_sa_diff.D1','ОФЗ_ОБР_РЕПО.D1','Monetization_coef.D1','total_output_gap','total_output_gap.D1','Unemployment_sa','Monetary_act','Monetary_unemp','ExpInfl_sa','Base_infl.D1','DebtGDP_log.D1','USDRUB.D1','Ruonia spot.D1','Unemployment_sa.D1']
ds1 = cut_ds1[['PC3','PC3.L1','PC3.D1']+regs_new].dropna()
# ds1=ds1.dropna()
ardl_model = ardl_data(ds1, 'PC3',['Ruonia spot','PMIservices','total_output_gap','Unemployment_sa','Monetary_act','ExpInfl_sa'], [], morder=3)
ardl_model.summary()
ds1['ECT'] = ardl_model.resid.shift(1)
ds1= ds1.dropna()
real_reg_lvl = sm.OLS(ds1['PC3.D1'], ds1[['ExpInfl_sa_diff.D1','ОФЗ_ОБР_РЕПО.D1','ECT']]).fit() #,'RateParity' #cov_type='HAC'
real_reg_lvl.summary()
real_reg_lvl.resid.hist()
(np.mean(real_reg_lvl.resid**2))**(1/2)

ds1['ОФЗ_ОБР_РЕПО.D1'].plot()
total_ds.loc['2014-09-01'][['Yld1','Yld15', 'PC2', 'Slope']]

plot_comparison(total_ds,'Urals','PC2')
ds = total_ds[['Slope', 'Urals','GDP.D1','total_output_sa','KeyRate','Total_infl','Unemployment_sa','total_output_sa.D1']].dropna()
plt.figure()
ax1 = plt.subplot(611)
ax1.plot(ds.index, ds['Slope'])
ax1.legend(['Slope'], loc="upper right")

ax2 = plt.subplot(612, sharex=ax1)
ax2.plot(ds.index, ds['Urals'])
ax2.legend(['Urals'], loc="upper right")

ax3 = plt.subplot(613, sharex=ax1)
ax3.plot(ds.index, ds['GDP.D1'])
ax3.legend(['GDP.D1'], loc="upper right")

ax4 = plt.subplot(614, sharex=ax1)
ax4.plot(ds.index, ds['KeyRate'])
ax4.legend(['KeyRate'], loc="upper right")

ax5 = plt.subplot(615, sharex=ax1)
ax5.plot(ds.index, ds['Total_infl'])
ax5.legend(['Total_infl'], loc="upper right")

ax6= plt.subplot(616, sharex=ax1)
ax6.plot(ds.index, ds['Unemployment_sa'])
ax6.legend(['Unemployment_sa'], loc="upper right")
total_ds['OFZ_Nonrez_%'].plot()


pc_corrmat_source = total_ds[(total_ds.index>'2004-12-01') & (total_ds.index<'2022-01-01')][cols]
pc_corrmat = pc_corrmat_source.corr()[['PC2','PC2.D1']]
del pc_corrmat_source
pc_corrmat.sort_values(by='PC2',inplace=True)

regs_new = ['Urals','Ruonia spot','total_output_gap','Urals','Total_infl','Ruonia spot.D1','Urals.D1','Total_infl.D1','Unemployment_sa.D1','PMIservices','ExpInfl_sa_diff.D1','ОФЗ_ОБР_РЕПО.D1','Monetization_coef.D1','total_output_gap','total_output_gap.D1','Unemployment_sa','Monetary_act','Monetary_unemp','ExpInfl_sa','Base_infl.D1','DebtGDP_log.D1','USDRUB.D1','Ruonia spot.D1','Unemployment_sa.D1']
ds1 = cut_ds1[['PC2','PC2.L1','PC2.D1']+regs_new].dropna()
ds1 = ds1[ds1.index<='2017-01-01']
# ds1=ds1.dropna()
ardl_model = ardl_data(ds1, 'PC2',['Ruonia spot','Urals','Total_infl','Unemployment_sa','ExpInfl_sa'], [], morder=2)
ardl_model.summary()

ds1['ECT'] = ardl_model.resid.shift(1)
ds1 = ds1.dropna()
real_reg_lvl = sm.OLS(ds1['PC2.D1'], ds1[['ECT']]).fit() #,'RateParity' #cov_type='HAC'
real_reg_lvl.summary()
real_reg_lvl.resid.hist()

build_regplots(ds1, ['Urals'], ['PC2'])
dsreg = ds1[['Urals','PC2']].dropna()
sns.regplot(dsreg, x='Urals', y='PC2')


monthly_data_long = pd.read_excel(disser_path + 'long_auctions.xlsx', index_col=0)
monthly_data_mid = pd.read_excel(disser_path + 'mid_auctions.xlsx', index_col=0)
monthly_data_short = pd.read_excel(disser_path + 'short_auctions.xlsx', index_col=0)
monthly_data_long['Demand to Allocation long'] = (monthly_data_long['Demand']/monthly_data_long['Allocation']).diff()
monthly_data_mid['Demand to Allocation mid'] = (monthly_data_mid['Demand']/monthly_data_mid['Allocation']).diff()


macroopros = pd.read_excel(papka_clean+'Macroopros.xlsx', index_col=0)
macroopros_monthly = macroopros.groupby(pd.Grouper(freq='MS')).first()
macroopros_monthly = macroopros_monthly.interpolate(method='polynomial', order=1)
macroopros_monthly['exp_growth'] = macroopros_monthly['GDP Y2']- macroopros_monthly['GDP Y0']
macroopros_monthly['exp_kr_change'] = macroopros_monthly['CBR Y1']- macroopros_monthly['CBR Y0']
macroopros_monthly_diff = macroopros_monthly.diff()
macroopros_monthly_diff['Demand to Allocation long'] = monthly_data_long['Demand to Allocation long'].fillna(0)
macroopros_monthly_diff['Demand to Allocation mid'] = monthly_data_mid['Demand to Allocation mid'].fillna(0)

gdpopros = pd.concat([total_ds[['PC1','PC1.D1','PC2','PC2.D1','PC2.L1','PC3','PC3.D1','PC3.L1','Total_infl']], macroopros_monthly_diff],axis=1).dropna()
implied_inflation_curve = pd.read_excel(papka_data+'Implied_Inflation_curve.xlsx', index_col=0)
gdpopros = pd.concat([gdpopros, implied_inflation_curve.diff()],axis=1).dropna()
gdpopros=gdpopros[gdpopros.index>='2023-02-01']
X = gdpopros[[4,'GDP Y0']]
Y = gdpopros['PC2.D1']
model = sm.OLS(Y, X).fit()
model.summary()
gdpopros['ECT'] = model.resid.shift()
gdpopros= gdpopros.dropna()
X = gdpopros[['Demand to Allocation long','ECT']]
Y = gdpopros['PC2.D1']
model = sm.OLS(Y, X).fit()
model.summary()
pd.concat([model.fittedvalues, Y],axis=1).plot()
gdpopros[['Demand to Allocation long','PC2.D1']].corr()
results_as_html = model.summary().tables[1].as_html()
ares = pd.read_html(results_as_html, header=0, index_col=0)[0]



X = sm.add_constant(gdpopros[['GDP Y1']])
Y = gdpopros['PC3']
model = sm.OLS(Y, X).fit()
model.summary()
gdpopros['ECT'] = model.resid.shift()
gdpopros= gdpopros.dropna()
X = gdpopros[['Demand to Allocation mid']]
Y = gdpopros['PC3.D1']
model = sm.OLS(Y, X).fit()
model.summary()

pd.concat([model.fittedvalues, Y],axis=1).plot()


corrmmat = gdpopros.corr()
corrmmat.to_excel(papka_tests+'corrs_dat.xlsx')






curve = cut_ds1[['Yld1','Yld3','Yld5','Yld10','Yld15']]
curve.columns = [1,3,5,10,15]
curve = curve[curve.index >= implied_inflation_curve.index[0]]

date = implied_inflation_curve.index[1]

curve.loc[date].diff().plot()
# curvediff = curve.diff()
implied_inflation_curve.diff().loc[date].plot()

from matplotlib.animation import FuncAnimation
# Set up the figure and axis
fig, ax = plt.subplots()
# Initialize the two line objects for the two curves
line1, = ax.plot([], [], label='Implied Inflation', color='b')
line2, = ax.plot([], [], label='Yield curve', color='r')
# Set up the labels, limits, and legend
ax.set_xlim(min(curve.columns), max(curve.columns))  # X-axis limits based on time to maturity
ax.set_ylim(min(curve.diff().min().min(), implied_inflation_curve.diff().min().min()), max(curve.diff().max().max(), implied_inflation_curve.diff().max().max()))  # Y-axis limits based on both DataFrames
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Yields')
ax.set_title('Yield Curves Over Time')
ax.legend()

# Initialize the background of the plot
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2
# Update function to animate both yield curves for each date
def update(frame):
    x1 = implied_inflation_curve.columns  # Time to maturity (X-axis)
    x2 = curve.columns
    y1 = implied_inflation_curve.iloc[frame]  # Yields for the first curve (Y-axis)
    y2 = curve.iloc[frame]  # Yields for the second curve (Y-axis)
    line1.set_data(x1, y1)
    line2.set_data(x2, y2)
    ax.set_title(f'Yield Curves diffs on {implied_inflation_curve.index[frame].strftime("%Y-%m-%d")}')
    return line1, line2
# Create animation with monthly frame iteration
ani = FuncAnimation(fig, update, frames=len(implied_inflation_curve), init_func=init, blit=False, interval=1000)
# Save animation as GIF
# ani.save(papka_plots+'yield_curves_animation_monthly.gif', writer='pillow')
plt.show()

ds_before22 = ds[ds.index<'2022-03-01']
ds_after22 = total_ds[total_ds.index>='2022-03-01'][['PC2','PC2.D1','PC2.L1','PC3','PC3.D1','PC3.L1','Urals.D1','DebtGDP_log.D1','Budget_balance_deseasoned.D1','PMIservices_log.D1','Balance_GDP.D1','Ruonia spot.D1','total_output_gap.D1','USDRUB.D1','Urals','Monetary_act.D1']]
ds_after22['Balance_GDP.D1'] = ds_after22['Balance_GDP.D1']*100
gdpopros = pd.concat([ds_after22, macroopros_monthly_diff],axis=1).dropna()
model = sm.tsa.AutoReg(gdpopros['PC2'], lags=1, trend='n',exog=gdpopros[['total_output_gap.D1','Urals']]).fit()
model = sm.OLS(gdpopros['PC2.D1'],exog=gdpopros[['Demand to Allocation long','exp_kr_change']]).fit()
model = sm.OLS(gdpopros['PC2'],exog=gdpopros[['total_output_gap.D1']]).fit()
model.summary()
pd.DataFrame(model.summary().tables[1]).to_excel(papka_tests+'2comp_lt.xlsx')
gdpopros['ECT'] = model.resid.shift()
gdpopros = gdpopros.dropna()
model2 = sm.OLS(gdpopros['PC2.D1'],exog=gdpopros[['Demand to Allocation mid','Balance_GDP.D1','ECT']]).fit()
model2.summary()
pd.DataFrame(model2.summary().tables[1]).to_excel(papka_tests+'2comp_st.xlsx')
model2.resid.hist()

pd.concat([gdpopros['PC2.D1'], model2.fittedvalues],axis=1).corr()
corr = gdpopros.corr()


model = sm.OLS(gdpopros['PC3'],exog=gdpopros[['total_output_gap.D1','Urals']]).fit()
model.summary()
pd.DataFrame(model.summary().tables[1]).to_excel(papka_tests+'2comp_дt.xlsx')
gdpopros['ECT'] = model.resid.shift()
gdpopros = gdpopros.dropna()
model2 = sm.OLS(gdpopros['PC3.D1'],exog=gdpopros[['Demand to Allocation mid','GDP Y2','ECT']]).fit(cov_type='HC1')
model2.summary()
pd.DataFrame(model2.summary().tables[1]).to_excel(papka_tests+'3comp_st.xlsx')
model2.resid.hist()
pd.concat([gdpopros['PC3.D1'], model2.fittedvalues],axis=1).plot()



