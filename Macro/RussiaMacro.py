import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy import stats

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
def hp_filter(dataset, name):
    output_total = dataset[[name]].dropna()
    cycle, trend = sm.tsa.filters.hpfilter(output_total, 129600)
    output_total["trend"] = trend
    output_total['output_gap'] = output_total[name] - output_total["trend"]
    output_total['output_gap_zscore'] = stats.zscore(output_total['output_gap'])
    return output_total

macrodata_path = '/Users/mmajidov/Projects/Disser/RussiaMacro/Data/'
clean_macrodata_path = '/Users/mmajidov/Projects/Disser/RussiaMacro/Data/clean_data/'


""" IVBO_ OKVED2_11-2024.xlsx
sheet 2 = Индекс выпуска товаров и услуг по базовым видам экономической деятельности 2013-2024гг.
(в постоянных ценах 2021 года, в % к предыдущему периоду)
sheet 1 = Индекс выпуска товаров и услуг по базовым видам экономической деятельности 2014-2024гг. 
(в постоянных ценах 2021 года, в % к соответствующему периоду предыдущего года)
"""
# для sheet 1
IVBO = pd.read_excel(macrodata_path+'IVBO_ OKVED2_11-2024.xlsx', sheet_name=1)
keywords = ["1 квартал", "2 квартал", "3 квартал", "4 квартал", "год"]
# filter rows where column title contains one of the keywords
IVBO = IVBO[~IVBO["Unnamed: 0"].isin(keywords)]
IVBO.index = list(range(1,13))
del IVBO["Unnamed: 0"]
output = []
date = []
for year in IVBO.columns:
    for month in IVBO.index:
        date.append(year[:-2]+'-'+str(month)+'-01')
        num = IVBO.loc[month][year]
        if type(num)==str:
            num = float(num.replace(',','.'))
        output.append(num)
output_df1424 = pd.DataFrame({'Output':output}, index = pd.to_datetime(date))
# output_df.plot()
# deseason
# output_deseasoned = seasonal_decompose(output_df.dropna(), model="additive", two_sided=False)
# output_df['output_deseasoned'] = output_deseasoned.trend

""" ИВБО_ОКВЭД_2007.xlsx
sheet 1 = Индекс выпуска товаров и услуг по базовым видам экономической деятельности
(в постоянных ценах 2011 года, в % к соответствующему периоду предыдущего года)
sheet 3 = Индекс выпуска товаров и услуг по базовым видам экономической деятельности 2004-2011гг. 
(в постоянных ценах 2008 года, в % к к соответствующему периоду предыдущего года)
"""
IVBO1116 = pd.read_excel(macrodata_path+'ИВБО_ОКВЭД_2007.xlsx', sheet_name=1)
keywords = ["1 квартал", "2 квартал", "3 квартал", "4 квартал", "год",]
# filter rows where column title contains one of the keywords
IVBO1116 = IVBO1116[~IVBO1116["Unnamed: 0"].isin(keywords)]
IVBO1116.index = list(range(1,13))
del IVBO1116["Unnamed: 0"]
output = []
date = []
for year in IVBO1116.columns:
    for month in IVBO1116.index:
        date.append(year[:-2]+'-'+str(month)+'-01')
        num = IVBO1116.loc[month][year]
        if type(num)==str:
            num = float(num.replace(',','.'))
        output.append(num)
output_df1116 = pd.DataFrame({'Output':output}, index = pd.to_datetime(date))
# output_df1116.plot()
# deseason
# output_deseasoned = seasonal_decompose(output_df.dropna(), model="additive", two_sided=False)
# output_deseasoned.trend
# output_df['output_deseasoned'] = output_deseasoned.trend


IVBO0411 = pd.read_excel(macrodata_path+'ИВБО_ОКВЭД_2007.xlsx', sheet_name=3)
keywords = ["1 квартал", "2 квартал", "3 квартал", "4 квартал", "год"]
# filter rows where column title contains one of the keywords
IVBO0411 = IVBO0411[~IVBO0411["Unnamed: 0"].isin(keywords)].dropna()
IVBO0411.index = list(range(1,13))
del IVBO0411["Unnamed: 0"]
output = []
date = []
for year in IVBO0411.columns:
    for month in IVBO0411.index:
        date.append(year[:-2]+'-'+str(month)+'-01')
        num = IVBO0411.loc[month][year]
        if type(num)==str:
            num = float(num.replace(',','.'))
        output.append(num)
output_df0411 = pd.DataFrame({'Output':output}, index = pd.to_datetime(date))
# output_df0411.plot()
# deseason
# output_deseasoned = seasonal_decompose(output_df.dropna(), model="additive", two_sided=False)
# output_deseasoned.trend
# output_df['output_deseasoned'] = output_deseasoned.trend

total_output = pd.concat([output_df1424, output_df1116[output_df1116.index<'2014-01-01']],axis=0)
total_output.sort_index(inplace=True)
total_output = pd.concat([total_output, output_df0411[output_df0411.index<'2012-01-01']],axis=0)
total_output.sort_index(inplace=True)
total_output = total_output.astype(float)
total_output = total_output/100-1
total_output.interpolate(method='polynomial', order=1,inplace=True)
# total_output.plot()
output_deseasoned = seasonal_decompose(total_output.dropna(), model="additive", two_sided=False)
total_output['output_deseasoned'] = output_deseasoned.trend

perform_stationarity_tests(total_output.dropna(), ['Output','output_deseasoned'])
total_output.to_excel(clean_macrodata_path+'total_output.xlsx')

output_data = pd.read_excel(clean_macrodata_path+'total_output.xlsx',index_col=0)
output_data_cumulative = (output_data+1).pow(1/12).cumprod()
output_filtered = hp_filter(output_data_cumulative, 'Output')
output_data['Output_cumulative'] = output_filtered['Output']
output_data['trend'] = output_filtered['trend']
output_data['output_gap'] = output_filtered['output_gap']
output_data['output_gap_zscore'] = output_filtered['output_gap_zscore']

# output_filtered[['Output','trend']].plot()
# output_filtered[['output_gap_zscore']].plot()

# output_data.to_excel(clean_macrodata_path+'total_output_gap.xlsx')
