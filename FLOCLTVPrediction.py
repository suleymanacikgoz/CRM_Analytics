
import pandas as pd
import numpy as np
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions


pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_csv("C:/Users/suley/OneDrive/Masaüstü/Miuul/Kodlarım/Odevler/3.Hafta/flo_data_20k.csv")

df=df_.copy()

def outlier_thresholds(dataframe, variable):
    quartilel1 = dataframe[variable].quantile(0.01)
    quartilel3 = dataframe[variable].quantile(0.99)
    interquantile_range=quartilel3-quartilel1
    up_limit=quartilel3+1.5*interquantile_range
    low_limit=quartilel1-1.5*interquantile_range
    #low_limit.round()
    #up_limit.round()
    return low_limit,up_limit


def replace_with_thresholds(dataframe,variable):
    low_limit,up_limit=outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable]=low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable]=up_limit


df.describe().T

replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")


df["total_order_sum"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["total_value_sum"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]



[df[col].dtype for col in df.columns]
df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])




df["last_order_date"].max()
today_date=dt.datetime(2021,6,1)

type(today_date)


cltv=pd.DataFrame()

cltv["recency_cltv_weekly"]=df["last_order_date"]-df["first_order_date"]
cltv["T_weekly"]=today_date-df["first_order_date"]
cltv["frequency"]=df["total_order_sum"]
cltv["monetary"]=df["total_value_sum"]

cltv.set_index(df.master_id, inplace=True)

cltv["monetary"]=cltv["monetary"]/cltv["frequency"]

cltv.describe().T

cltv["recency_cltv_weekly"]=cltv["recency_cltv_weekly"]/7

cltv["T_weekly"]=cltv["T_weekly"]/7

cltv['recency_cltv_weekly'] = cltv['recency_cltv_weekly'].apply(lambda x: x.days)
cltv['T_weekly'] = cltv['T_weekly'].apply(lambda x: x.days)

cltv=cltv[(cltv["recency_cltv_weekly"]>0)]
cltv=cltv[(cltv["T_weekly"]>0)]


bgf=BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])

cltv.describe().T


bgf.conditional_expected_number_of_purchases_up_to_time(1,cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"]).sort_values(ascending=False).head(10)



cltv["exp_sales_3_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(12,cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])


cltv["exp_sales_6_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(24,cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])


plot_period_transactions(bgf)


