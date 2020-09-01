import pandas as pd
import numpy as np


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

    delist_df = raw_df[raw_df['DLSTCD'].notnull()].copy().reset_index(drop=True)

    # Choose the delisting code and assign  -0.3 to original df (1997 Shumway) Note 9
    delist_500 = delist_df[(delist_df['DLSTCD'] == 500) | (delist_df['DLSTCD'] == 520) | (
                (delist_df['DLSTCD'] >= 551) & (delist_df['DLSTCD'] <= 573)) | (delist_df['DLSTCD'] == 574) | (
                                       delist_df['DLSTCD'] == 584)]
    delist_df.loc[delist_500['DLRET'].index.tolist(), 'DLRET'] = -0.3
    del delist_500
    # Copy the RET to DLRET for stocks delisted by above code
    delist_df.loc[delist_df['DLRET'] == 'S', 'DLRET'] = delist_df.loc[delist_df['DLRET'] == 'S', 'RET']
    if len(delist_df.loc[delist_df['DLRET'] == 'S', 'DLRET']) != 0:
        raise Exception('The DLRET with value S is not completely cleaned')
    else:
        print('The DLRET with value S is cleaned')
    delist_df.loc[delist_df['DLRET'] == 'T', 'DLRET'] = delist_df.loc[delist_df['DLRET'] == 'T', 'RET']
    if len(delist_df.loc[delist_df['DLRET'] == 'T', 'DLRET']) != 0:
        raise Exception('The DLRET with value T is not completely cleaned')
    else:
        print('The DLRET with value T is cleaned')
    delist_df.loc[delist_df['DLRET'] == 'A', 'DLRET'] = delist_df.loc[delist_df['DLRET'] == 'A', 'RET']
    if len(delist_df.loc[delist_df['DLRET'] == 'A', 'DLRET']) != 0:
        raise Exception('The DLRET with value A is not completely cleaned')
    else:
        print('The DLRET with value A is cleaned')
    delist_df.loc[delist_df['DLRET'] == 'P', 'DLRET'] = delist_df.loc[delist_df['DLRET'] == 'P', 'RET']
    if len(delist_df.loc[delist_df['DLRET'] == 'P', 'DLRET']) != 0:
        raise Exception('The DLRET with value P is not completely cleaned')
    else:
        print('The DLRET with value P is cleaned')
    # replace raw dara RET with DLRET for delisted stocks.
    raw_df.loc[delist_df.index, 'RET'] = delist_df['DLRET']

    # delete the observations where ret and vol is null and prc is negative(which means there is no closing price data)
    df_notnull = raw_df[(raw_df['RET'].notnull()) & (raw_df['VOL'].notnull()) &
                        (raw_df['RET'] != 'A') & (raw_df['RET'] != 'B') & (raw_df['RET'] != 'C') &
                        (raw_df['RET'] != 'D') & (raw_df['RET'] != 'E')].reset_index(drop=True)
    print("The number of obs with RET and VOL is", len(df_notnull), "The raw df obs is", len(raw_df))
    del raw_df
    # Amihud selection criteria i to iii.
    df_notnull['year'] = df_notnull['date'].apply(lambda x: str(x)[0:4])

    # Step i
    # stocks less than 200 trading days each year
    df_counts = df_notnull.groupby(['year', 'PERMNO']).size().reset_index(name='counts')
    df_trading_days_200 = df_counts[df_counts['counts'] > 200][['year', 'PERMNO']]
    """
    # stocks delisted in this year
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
    df_compute_amihud = df_notnull[(df_notnull['VOL'] != 0) & (df_notnull['PRC'] > 0)].reset_index(drop=True)
    df_compute_amihud['dollar_vol'] = df_compute_amihud['PRC'] * df_compute_amihud['VOL']
    df_compute_amihud['amihud_d'] = abs(df_compute_amihud['RET'].astype(np.float64)) / df_compute_amihud['dollar_vol']
    df_div_amihud = df_compute_amihud.groupby(['year', 'PERMNO']).sum().reset_index()[['year', 'PERMNO', 'amihud_d', 'DIVAMT']]
    df_div_amihud['counts'] = df_compute_amihud.groupby(['year', 'PERMNO']).size().reset_index(name='counts')['counts']
    df_step1 = df_trading_days_200.merge(df_div_amihud, 'left', on=['year', 'PERMNO'])

    """
    df_notnull['dollar_vol'] = df_notnull['PRC'] * df_notnull['VOL']
    df_notnull['amihud_d'] = abs(df_notnull['RET'].astype(np.float64)) / df_notnull['dollar_vol']
    df_div_amihud = df_notnull.groupby(['year', 'PERMNO']).sum().reset_index()[['year', 'PERMNO', 'amihud_d', 'DIVAMT']]
    """
    # step ii and iii
    # stocks less than 5 dollors at the year end shrout not equal to zero
    df_last = df_notnull.groupby(['year', 'PERMNO']).last().reset_index()
    df_step23 = df_last[(abs(df_last['PRC']) > 5) & (df_last['SHROUT'] != 0)].reset_index(drop=True)[['PERMNO', 'year', 'PRC', 'SHROUT']]
    df_step123 = df_step1.merge(df_step23, 'inner', on=['PERMNO', 'year'])
    # calculate annual based variables
    df_step123['yield'] = (df_step123['DIVAMT']) / abs(df_step123['PRC']) * 100
    df_step123['amihud_y'] = (df_step123['amihud_d'] / df_step123['counts']) * 1000000
    df_step123['size'] = abs((df_step123['PRC']) * df_step123['SHROUT']) / 1000
    # iv step deletion
    print("after step123", df_step123.groupby(['year']).size().reset_index(name='counts'))
    def delete_amihud(x):
        shres_01 = x['amihud_y'].quantile(0.01, interpolation='linear')
        shres_99 = x['amihud_y'].quantile(0.99, interpolation='linear')
        return x[(x['amihud_y'] > shres_01) & (x['amihud_y'] < shres_99)].reset_index(drop=True)
    final_df = df_step123.groupby(['year']).apply(lambda x: delete_amihud(x)).reset_index(drop=True)
    print(final_df)
    print("after step4", final_df.groupby(['year']).size().reset_index(name='counts'))

    
    final_df.to_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\testtb1.csv', index=False)
    # stock counts
    print("final counts", final_df.groupby(['year']).size().reset_index(name='counts'))
    # annual means and std
    print("mean", final_df.groupby(['year']).mean().reset_index()[['amihud_y', 'yield', 'size']])
    print("std", final_df.groupby(['year']).std().reset_index()[['amihud_y', 'yield', 'size']])
    # mean of annual mean
    print(np.mean(final_df.groupby(['year']).mean().reset_index()['amihud_y']))
    print(np.mean(final_df.groupby(['year']).mean().reset_index()['yield']))
    print(np.mean(final_df.groupby(['year']).mean().reset_index()['size']))
    # mean of annual std
    print(np.mean(final_df.groupby(['year']).std().reset_index()['amihud_y']))
    print(np.mean(final_df.groupby(['year']).std().reset_index()['yield']))
    print(np.mean(final_df.groupby(['year']).std().reset_index()['size']))


if __name__ == '__main__':
    raw_df = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\amihud_raw_data.csv')
    df_step123 = data_cleaning(raw_df)
