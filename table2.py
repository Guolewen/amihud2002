import pandas as pd
import numpy as np
from appelpy.linear_model import OLS

def clean_monthly_data(monthly_df):
    """

    :param monthly_df: pandas df of raw data from CRSP monthly securty files, only NYSE stocks are included
    :return: pandas df of cleaned and delist-adjusted return data:
    1. The first month return of a newly IPO stock is deleted.
    2. The returns with error code 'B' is deleted
    3. Assign -0.3 to original delisted return for stocks where the delisting code as 1997 Shumway (Note 9)
    4. Replace the last returns with approriate delisted returns.
    """
    monthly_df['year'] = monthly_df['date'].apply(lambda x: str(x)[0:4])
    delist_500 = monthly_df[(monthly_df['DLSTCD'] == 500) | (monthly_df['DLSTCD'] == 520) | (
            (monthly_df['DLSTCD'] >= 551) & (monthly_df['DLSTCD'] <= 573)) | (monthly_df['DLSTCD'] == 574) | (
                                   monthly_df['DLSTCD'] == 584)]
    monthly_df.loc[delist_500['DLRET'].index.tolist(), 'DLRET'] = -0.3
    del delist_500
    
    def delete_first_month_ret(x):
        if x.empty:
            return None
        elif (x.iloc[0]['RET'] == 'C') or (x.iloc[0]['RET'] == 'B') or (pd.isna(x.iloc[0]['RET'])):
            return delete_first_month_ret(x.iloc[1:])
        else:
            return x

    def delete_last_month_ret(x):
        # delete the end data with ret is 'B' first:
        if (x.iloc[-1]['RET'] == 'B') or ((pd.isna(x.iloc[-1]['RET'])) and (pd.isna(x.iloc[-2]['DLSTCD'])) and (pd.isna(x.iloc[-1]['DLSTCD']))):
            return delete_last_month_ret(x.iloc[:-1])
        elif (not pd.isna(x.iloc[-1]['DLSTCD'])) and (pd.isna(x.iloc[-1]['RET'])):
            x.iloc[-1, x.columns.get_loc('RET')] = x.iloc[-1]['DLRET']
            return x
        elif (pd.isna(x.iloc[-1]['DLSTCD'])) and (pd.isna(x.iloc[-1]['RET'])):
            if (not pd.isna(x.iloc[-2]['DLSTCD'])) and (not pd.isna(x.iloc[-2]['RET'])):
                x.iloc[-1, x.columns.get_loc('RET')] = x.iloc[-2]['DLRET']
                return x
            elif (not pd.isna(x.iloc[-2]['DLSTCD'])) and (pd.isna(x.iloc[-2]['RET'])):
                x.iloc[-2, x.columns.get_loc('RET')] = x.iloc[-2]['DLRET']
                return x.iloc[:-1]
        else:
            return x
    first_month_cleaned = monthly_df.groupby(['PERMNO']).apply(lambda x: delete_first_month_ret(x)).reset_index(drop=True)
    last_cleaned = first_month_cleaned.groupby(['PERMNO']).apply(lambda x: delete_last_month_ret(x)).reset_index(drop=True)
    cleaned = last_cleaned[
        (last_cleaned['RET'] != 'C') & (last_cleaned['RET'] != 'B') & (last_cleaned['RET'].notnull())].reset_index(
        drop=True)
    cleaned['RET'] = cleaned['RET'].astype(np.float64) * 100
    return cleaned


def convert_stock_char_for_merge(stock_char):
    stock_char['year'] = stock_char['year'].apply(lambda x: str(int(x) + 1))
    return stock_char

def reg_results(merged_df):
    merged_df['LogSize'] = np.log(merged_df['size'])
    y_list = ['RET']
    x1_list = ['Beta', 'ILLQMA', 'R100', 'R100YR']
    x2_list = ['Beta', 'ILLQMA', 'R100', 'R100YR', 'LogSize', 'sdret', 'yield']

    def reg_model1(x):
        reg_model1 = OLS(x, y_list, x1_list).fit()
        return reg_model1.results.params.to_frame().transpose()

    def reg_model2(x):
        reg_model2 = OLS(x, y_list, x2_list).fit()
        return reg_model2.results.params.to_frame().transpose()



    coefficients_df_model1 = merged_df.groupby(['date']).apply(lambda x: reg_model1(x)).reset_index()
    means_model1 = coefficients_df_model1.mean().to_frame(name='coef_model1')[2:]
    se_model1 = coefficients_df_model1.sem().to_frame(name='se_model1')[2:]
    output_df_model1 = pd.concat([means_model1, se_model1], axis=1)
    output_df_model1['t_model1'] = output_df_model1['coef_model1'] / output_df_model1['se_model1']

    coefficients_df_model2 = merged_df.groupby(['date']).apply(lambda x: reg_model2(x)).reset_index()
    means_model2 = coefficients_df_model2.mean().to_frame(name='coef_model2')[2:]
    se_model2 = coefficients_df_model2.sem().to_frame(name='se_model2')[2:]
    output_df_model2 = pd.concat([means_model2, se_model2], axis=1)
    output_df_model2['t_model2'] = output_df_model2['coef_model2'] / output_df_model2['se_model2']
    final_output = pd.concat([output_df_model1, output_df_model2], axis=1)

    return final_output


if __name__ == '__main__':

    monthly_df = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\stock_month_1964to1997.csv')
    stock_char = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\stock_char_1963to1996.csv')
    stock_char_converted = convert_stock_char_for_merge(stock_char)
    ret_cleaned = clean_monthly_data(monthly_df)
    merged_df = ret_cleaned.merge(stock_char_converted, 'inner', on=['year', 'PERMNO'])
    merged_df.to_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\merged_df.csv')
    final_output = reg_results(merged_df)
    print(final_output)
    """
    comparision
                                                                                        Amihud 2002
           coef_allmonths  t_allmonths  coef_allmonths   t_allmonths          
    const          0.872467    (1.672033)     1.427928   (3.698525)         -0.364   (0.76)       1.922   (4.06)
    Beta           0.104768    (0.236903)     0.366779   (1.279470)          1.183   (2.45)       0.217   (0.64)
    ILLQMA(KeyVar) 0.126585    (4.880535)     0.093907   (4.209741)          0.162   (6.55)       0.112   (5.39)
    R100           0.769623    (3.245838)     0.708398   (3.473633)          1.023   (3.83)       0.888   (3.70)
    R100YR         0.426061    (2.975364)     0.366015   (3.291667)          0.382   (2.98)       0.359   (3.40)
    LogSize          NaN         NaN         -0.062105   (-1.680818)                             -0.134   (-3.50)
    sdret            NaN         NaN         -0.192250   (-2.014094)                             -0.179   (-1.90)
    yield            NaN         NaN         -0.012916   (-0.881601)                             -0.048   (-3.36)
    
            Larry Harris & Andrea Amato(2016 Replication)
                    -0.574      (1.29)         1.627     (3.25)
                     1.322      (2.97)         0.347     (1.02)
                     0.112      (4.16)         0.071     (3.35)
                     1.076      (4.07)         1.012     (4.38)
                     0.41       (2.59)         0.38      (2.87)
                                               -0.100    (-2.54)
                                               -0.249    (-2.74)
                                               -0.028    (-1.55)
    """