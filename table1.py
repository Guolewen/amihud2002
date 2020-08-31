import pandas as pd
import numpy as np


def data_cleaning(raw_df):
    """
    We first adjust delisting return : line 7 to line 35

    Then delete the data with volome and return not available and the closing price negative
    and then compute daily amihud: line 37 to line 44

    Amihud selection criteria I: The stock has return and volume data for more than 200 days during year
    y-1: This makes the estimated parameters more reliable. Also, the stock must be listed at the end of year y-1:
    """
    delist_df = raw_df[raw_df['DLSTCD'].notnull()].copy()
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
    df_notnull = raw_df[(raw_df['RET'].notnull()) & (raw_df['VOL'] >= 0) &
                        (raw_df['RET'] != 'A') & (raw_df['RET'] != 'B') & (raw_df['RET'] != 'C') &
                        (raw_df['RET'] != 'D') & (raw_df['RET'] != 'E')].reset_index(drop=True)
    print("The number of obs with RET and VOL is", len(df_notnull), "The raw df obs is", len(raw_df))
    del raw_df
    # Amihud selection criteria I: exclude stocks with less than 200 trading days and stock
    df_notnull['year'] = df_notnull['date'].apply(lambda x: str(x)[0:4])
    # stocks less than 200 trading days each year
    df_counts = df_notnull.groupby(['year', 'PERMNO']).size().reset_index(name='counts')
    group_counts = df_counts[df_counts['counts'] <= 200].groupby(['year'])
    # stocks delisted in this year
    delist_df['year'] = delist_df['date'].apply(lambda x: str(x)[0:4])
    group_delist = delist_df.groupby(['year'])
    # stocks less than 5 dollors at the year end
    df_last = df_notnull.groupby(['year', 'PERMNO']).last().reset_index()
    group_p_smaller5 = df_last[df_last['PRC'] <= 5].groupby(['year'])
    df_last['MarketCap'] = df_last['PRC'] * df_last['SHROUT']
    # store all the stocks which need to be exclude for each year for step i to iii
    stocks_delete = {}
    for year in group_delist.groups.keys():
        try:
            set_delist = set(group_delist.get_group(year)['PERMNO'])
        except KeyError:
            set_delist = set()
        try:
            set_counts = set(group_counts.get_group(year)['PERMNO'])
        except KeyError:
            set_counts = set()
        try:
            set_smaller5 = set(group_p_smaller5.get_group(year)['PERMNO'])
        except KeyError:
            set_smaller5 = set()
        final_list = list(set_counts.union(set_delist).union(set_smaller5))
        stocks_delete.update({year: final_list})
    print(stocks_delete)
    # calculate the daily amihud, if vol is 0 then exclude the daily observation for that stock.
    df_notnull = df_notnull[df_notnull['VOL'] > 0].reset_index()
    df_notnull['dollar_vol'] = df_notnull['PRC'] * df_notnull['VOL']
    df_notnull['amihud_d'] = abs(df_notnull['RET'].astype(np.float64)) / df_notnull['dollar_vol']
    # delete stocks and form the final dataset
    df_div_amihud = df_notnull.groupby(['year', 'PERMNO']).sum().reset_index()[['year', 'PERMNO', 'amihud_d', 'DIVAMT']]
    df_div_amihud['yield'] = (df_div_amihud['DIVAMT'] / df_last['PRC']) * 100
    df_div_amihud['amihud_y'] = (df_div_amihud['amihud_d'] / df_counts['counts']) * 1000000
    df_div_amihud['size'] = df_last['MarketCap']

    def delete_stock(x, stocks_delete):
        year = list(set(x['year']))[0]
        return x[~x.PERMNO.isin(stocks_delete[year])].reset_index(drop=True)
    df_deleted_1to3 = df_div_amihud.groupby(['year']).apply(lambda x: delete_stock(x, stocks_delete)).reset_index(drop=True)
    print(df_deleted_1to3)
    print("after step1to3", df_deleted_1to3.groupby(['year']).size().reset_index(name='counts'))
    # iv step deletion
    group_amihud = df_deleted_1to3.groupby(['year'])
    step4_deletion = {}
    for year in group_delist.groups.keys():
        delete_index = np.where(pd.qcut(group_amihud.get_group(year)['amihud_y'], q=[0.01, 0.99]).isna())[0]
        set_amihud = set(group_amihud.get_group(year).reset_index().loc[delete_index]['PERMNO'])
        amihud_del_list = list(set_amihud)
        step4_deletion.update({year: amihud_del_list})
    print("step4 deletion", step4_deletion)
    final_df = df_deleted_1to3.groupby(['year']).apply(lambda x: delete_stock(x, step4_deletion)).reset_index(drop=True)
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

raw_df = pd.read_csv(r'D:\wooldride_econometrics\paper_liq_replicating\amihud_2002\amihud_raw_data.csv')
data_cleaning(raw_df)