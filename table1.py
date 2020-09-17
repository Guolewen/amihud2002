import pandas as pd
import numpy as np
from appelpy.linear_model import OLS


def delete_stock(x, stocks_delete):
    year = list(set(x['year']))[0]
    return x[~x.PERMNO.isin(stocks_delete[year])].reset_index(drop=True)


def data_cleaning(raw_df):
    """
    We first adjust delisting return : line 7 to line 35

    Then delete the data with volome and return not available and the closing price negative
    and then compute daily amihud: line 37 to line 44

    Amihud selection criteria I: The stock has return and volume data for more than 200 days during year
    y-1: This makes the estimated parameters more reliable. Also, the stock must be listed at the end of year y-1:
    """
    # delete ADRs
    #raw_df['SHRCD1'] = raw_df['SHRCD'].apply(lambda x: str(x)[0])
    # raw_df = raw_df[raw_df['SHRCD1'] != '3'].reset_index()
    # delete special dividend
    raw_df['DISTCD_1'] = raw_df['DISTCD'].apply(lambda x: str(x)[0])
    raw_df["DISTCD_3"] = raw_df['DISTCD'].apply(lambda x: str(x)[2])
    # not cash divdend
    no_cash_div = raw_df[(raw_df['FACPR'] != 0) & (raw_df['DISTCD_1'] != '6')]
    raw_df.loc[no_cash_div.index, 'DIVAMT'] = 0
    #  an offer price given to a certain amount of shares adjustment
    adjusted_div_6 = (raw_df[(raw_df['DISTCD_1'] == '6') & (raw_df['FACPR'] < 0)]['DIVAMT']) * abs(
                      raw_df[(raw_df['DISTCD_1'] == '6') & (raw_df['FACPR'] < 0)]['FACPR'])
    raw_df.loc[adjusted_div_6.index, 'DIVAMT'] = adjusted_div_6.tolist()

    delist_df = raw_df[raw_df['DLSTCD'].notnull()].copy().reset_index(drop=True)
    # delete the observations where ret and vol is null and prc is negative(which means there is no closing price data)
    df_notnull = raw_df[(raw_df['RET'].notnull()) & (raw_df['VOL'] > 0) &
                        (raw_df['RET'] != 'A') & (raw_df['RET'] != 'B') & (raw_df['RET'] != 'C') &
                        (raw_df['RET'] != 'D') & (raw_df['RET'] != 'E')].reset_index(drop=True)
    print("The number of obs with RET and VOL is", len(df_notnull), "The raw df obs is", len(raw_df))
    del raw_df
    # Amihud selection criteria i to iii.
    df_notnull['year'] = df_notnull['date'].apply(lambda x: str(x)[0:4])
    df_notnull['RET'] = df_notnull['RET'].astype(np.float64)
    df_notnull['dollar_vol'] = df_notnull['PRC'] * df_notnull['VOL']
    df_notnull['amihud_d'] = abs(df_notnull['RET']) / df_notnull['dollar_vol']
    # Step i
    # stocks less than 200 trading days each year
    # stocks delisted in this year
    """
    delist_df['year'] = delist_df['date'].apply(lambda x: str(x)[0:4])
    group_delist = delist_df.groupby(['year'])
    stocks_delist = {}
    for year in group_delist.groups.keys():
        set_delist = set(group_delist.get_group(year)['PERMNO'])
        final_list = list(set_delist)
        stocks_delist.update({year: final_list})
    df_delistandtradingdays = df_trading_days_200.groupby(['year']).apply(lambda x: delete_stock(x, stocks_delist)).reset_index(drop=True)
    """
    # the following delete vol equal 0 and price is negative
    # df_notnull = df_notnull[(df_notnull['VOL'] != 0) & (df_notnull['PRC'] > 0)].reset_index(drop=True)
    df_counts = df_notnull.groupby(['year', 'PERMNO']).size().reset_index(name='counts')
    df_trading_days_200 = df_counts[df_counts['counts'] > 200][['year', 'PERMNO']].reset_index(drop=True)
    df_div_amihud = df_notnull.groupby(['year', 'PERMNO']).sum().reset_index()[['year', 'PERMNO', 'amihud_d', 'DIVAMT']]
    df_div_amihud['counts'] = df_notnull.groupby(['year', 'PERMNO']).size().reset_index(name='counts')['counts']
    df_div_amihud['sdret'] = df_notnull.groupby(['year', 'PERMNO']).std().reset_index()['RET'] * 100
    df_step1 = df_trading_days_200.merge(df_div_amihud, 'left', on=['year', 'PERMNO'])
    # step ii and iii
    # stocks less than 5 dollors at the year end shrout not equal to zero
    df_last = df_notnull.groupby(['year', 'PERMNO']).last().reset_index()
    df_step23 = df_last[((abs(df_last['PRC'])) > 5) & (df_last['SHROUT'] != 0)].reset_index(drop=True)[['PERMNO', 'year', 'PRC', 'SHROUT']]
    df_step123 = df_step1.merge(df_step23, 'inner', on=['PERMNO', 'year'])
    del df_last, df_step23, df_step1
    # calculate annual based variables
    df_step123['yield'] = (df_step123['DIVAMT']) / abs(df_step123['PRC']) * 100
    df_step123['amihud_y'] = (df_step123['amihud_d'] / df_step123['counts']) * 1000000
    df_step123['size'] = (df_step123['PRC']) * df_step123['SHROUT'] / 1000
    # iv step deletion
    print("after step123", df_step123.groupby(['year']).size().reset_index(name='counts'))
    def delete_amihud(x):
        shres_01 = x['amihud_y'].quantile(0.01, interpolation='higher')
        shres_99 = x['amihud_y'].quantile(0.99, interpolation='lower')
        return x[(x['amihud_y'] > shres_01) & (x['amihud_y'] < shres_99)].reset_index(drop=True)
    final_df = df_step123.groupby(['year']).apply(lambda x: delete_amihud(x)).reset_index(drop=True)

    # get addtional variables
    amihud_y = final_df.groupby(['year']).mean().reset_index()[['year', 'amihud_y']]
    amihud_y.rename(columns={'amihud_y': 'Avg_amihud'}, inplace=True)
    final_df = final_df.merge(amihud_y, how='left', on='year')
    final_df['ILLQMA'] = final_df['amihud_y'] / final_df['Avg_amihud']
    def retfirst100days(x):
        cleaned_df = x.drop_duplicates(subset=['date'])
        first_ret_price_array = abs(cleaned_df.iloc[[0, -101]]['PRC']).values
        last_ret_price_array = abs(cleaned_df.iloc[[-100, -1]]['PRC']).values
        return (first_ret_price_array[1] / first_ret_price_array[0]) - 1, (last_ret_price_array[1] / last_ret_price_array[0]) - 1
    # calculate R100YR and R100
    df_1 = df_trading_days_200.merge(df_notnull, 'left', on=['year', 'PERMNO'])
    df_1 = df_1.groupby(['year', 'PERMNO']).apply(lambda x: retfirst100days(x)).reset_index(name='Ret100')
    df_1[['R100YR', 'R100']] = pd.DataFrame(df_1['Ret100'].tolist(), index=df_1.index)
    final_df = final_df.merge(df_1, 'left', on=['year', 'PERMNO'])
    # stock counts
    print("final counts", final_df.groupby(['year']).size().reset_index(name='counts'))
    # annual means and std
    print("mean", final_df.groupby(['year']).mean()[['amihud_y', 'yield', 'size', 'sdret']])
    print("std", final_df.groupby(['year']).std()[['amihud_y', 'yield', 'size', 'sdret']])
    print("skewess", final_df.groupby(['year']).skew()[['amihud_y', 'yield', 'size', 'sdret']])
    # mean of annual mean
    print("mean of annual mean")
    print(np.mean(final_df.groupby(['year']).mean().reset_index()['amihud_y']))
    print(np.mean(final_df.groupby(['year']).mean().reset_index()['yield']))
    print(np.mean(final_df.groupby(['year']).mean().reset_index()['size']))
    print(np.mean(final_df.groupby(['year']).mean().reset_index()['sdret']))
    # mean of annual std
    print("mean of annual std")
    print(np.mean(final_df.groupby(['year']).std().reset_index()['amihud_y']))
    print(np.mean(final_df.groupby(['year']).std().reset_index()['yield']))
    print(np.mean(final_df.groupby(['year']).std().reset_index()['size']))
    print(np.mean(final_df.groupby(['year']).std().reset_index()['sdret']))
    # median of annual skew
    print("mean of annual skew")
    print(np.mean(final_df.groupby(['year']).skew().reset_index()['amihud_y']))
    print(np.mean(final_df.groupby(['year']).skew().reset_index()['yield']))
    print(np.mean(final_df.groupby(['year']).skew().reset_index()['size']))
    print(np.mean(final_df.groupby(['year']).skew().reset_index()['sdret']))
    return final_df

def calculate_beta(raw_df, index_df):
    index_df.rename(columns={'caldt': 'date'}, inplace=True)
    index_df['year'] = index_df['date'].apply(lambda x: str(x)[0:4])
    df_notnull = raw_df[(raw_df['RET'].notnull()) & (raw_df['VOL'] > 0) &
                        (raw_df['RET'] != 'A') & (raw_df['RET'] != 'B') & (raw_df['RET'] != 'C') &
                        (raw_df['RET'] != 'D') & (raw_df['RET'] != 'E')].reset_index(drop=True)
    df_notnull['year'] = df_notnull['date'].apply(lambda x: str(x)[0:4])
    df_notnull['RET'] = df_notnull['RET'].astype(np.float64)
    del raw_df
    df_last = df_notnull.groupby(['year', 'PERMNO']).last().reset_index()
    df_last['size'] = abs(df_last['PRC']) * df_last['SHROUT']

    def sort_portfolios(x):
        my_dict = {}
        q_cut = x['size'].quantile(q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        my_dict.update({'0.1': x[x['size'] < q_cut[0.1]]['PERMNO'].tolist()})
        my_dict.update({'0.2': x[(x['size'] >= q_cut[0.1]) & (x['size'] < q_cut[0.2])]['PERMNO'].tolist()})
        my_dict.update({'0.3': x[(x['size'] >= q_cut[0.2]) & (x['size'] < q_cut[0.3])]['PERMNO'].tolist()})
        my_dict.update({'0.4': x[(x['size'] >= q_cut[0.3]) & (x['size'] < q_cut[0.4])]['PERMNO'].tolist()})
        my_dict.update({'0.5': x[(x['size'] >= q_cut[0.4]) & (x['size'] < q_cut[0.5])]['PERMNO'].tolist()})
        my_dict.update({'0.6': x[(x['size'] >= q_cut[0.5]) & (x['size'] < q_cut[0.6])]['PERMNO'].tolist()})
        my_dict.update({'0.7': x[(x['size'] >= q_cut[0.6]) & (x['size'] < q_cut[0.7])]['PERMNO'].tolist()})
        my_dict.update({'0.8': x[(x['size'] >= q_cut[0.7]) & (x['size'] < q_cut[0.8])]['PERMNO'].tolist()})
        my_dict.update({'0.9': x[(x['size'] >= q_cut[0.8]) & (x['size'] < q_cut[0.9])]['PERMNO'].tolist()})
        my_dict.update({'1.0': x[(x['size'] >= q_cut[0.9])]['PERMNO'].tolist()})
        return my_dict

    df_t = df_last.groupby(['year']).apply(lambda x: sort_portfolios(x)).reset_index()
    data_list_dict = [i for i in df_t[0]]
    year_list = [year for year in df_t['year']]
    frames = []
    for dict, year in zip(data_list_dict, year_list):
        unmelt_df = pd.DataFrame.from_dict(dict, orient='index').transpose()
        unmelt_df['year'] = year
        melted_df = pd.melt(unmelt_df, id_vars=['year'],
                            value_vars=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                            var_name='port_ind',
                            value_name='PERMNO')
        melted_df = melted_df.dropna(subset=['PERMNO']).reset_index(drop=True)
        melted_df['PERMNO'] = melted_df['PERMNO'].astype(np.int64)
        frames.append(melted_df)
    final_df = df_notnull[['PERMNO', 'date', 'year', 'RET']].merge(pd.concat(frames, ignore_index=True),
                                                                   how='left', on=['year', 'PERMNO'])
    ewret_df = final_df.groupby(['date', 'port_ind']).mean()['RET'].unstack(level=-1).reset_index()
    reg_df = index_df.merge(ewret_df, 'left', on='date')

    def reg_beta(x):
        y_list = ['ewretd']
        x_list_of_lists = [['0.1'], ['0.2'], ['0.3'], ['0.4'], ['0.5'], ['0.6'], ['0.7'], ['0.8'], ['0.9'], ['1.0']]
        a_dict = {}
        for x_list in x_list_of_lists:
            a_dict.update({OLS(x, y_list, x_list).fit().results_output_standardized.data.index[0]: OLS(x, y_list, x_list).fit().results_output_standardized.data.loc[x_list[0], 'coef']})
        return pd.DataFrame.from_dict([a_dict])
    beta_df = reg_df.groupby(['year']).apply(lambda x: reg_beta(x)).reset_index()
    stock_assign_beta =pd.melt(beta_df, id_vars=['year'],
                               value_vars=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                               var_name='port_ind',
                               value_name='Beta').merge(pd.concat(frames, ignore_index=True), 'right',
                                                        on=['year', 'port_ind'])

    return stock_assign_beta

if __name__ == '__main__':
    raw_df = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\amihud_raw_data.csv')
    index_df = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\index_return.csv')
    df_stock_char = data_cleaning(raw_df)
    df_stock_assign_beta = calculate_beta(raw_df, index_df)
    stock_char_tb2_1963to1996 = df_stock_char.merge(df_stock_assign_beta, 'left', on=['year', 'PERMNO'])
    stock_char_tb2_1963to1996.to_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\stock_char_1963to1996.csv', index=False)
    """
    Comparisions
    mean of annual means                  Amihud(2002)               Larry Harris & Andrea Amato(2016 Replication)
    ILLIQ  0.27526723613781107             0.337                               0.282
    SIZE   870.806768861622                792.6                               868.2
    DIVYLD 4.163978375230819               4.14                                3.90
    SDRET  2.01013085388389                2.08                                2.02
    mean of annual std
    ILLIQ  0.39415578306742666             0.512                               0.404
    SIZE   1682.9718436950504              1611.5                              1710.7
    DIVYLD 6.511587181643278               5.48                                5.48
    SDRET  0.7211605561022127              0.75                                0.73
    mean of annual skew
    ILLIQ  2.915427476872194               3.095                               2.945
    SIZE   5.061406345756672               5.417                               5.219
    DIVYLD 8.274584483201552               5.385                               6.129
    SDRET  1.067488504462343               1.026                               1.035
    
    
    """
