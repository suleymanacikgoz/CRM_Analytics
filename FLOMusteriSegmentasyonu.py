

import pandas as pd
import numpy as np
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_csv("C:/Users/suley/OneDrive/Masaüstü/Miuul/Kodlarım/Odevler/3.Hafta/flo_data_20k.csv")

df=df_.copy()



          
                    
df.head(10)
                     
df.columns
                    
df.describe().T
                     
df.isnull().sum()
                     
[df[col].dtype for col in df.columns]
           
df["total_order_sum"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["total_value_sum"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]


          

[df[col].dtype for col in df.columns]
df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])

           

df.groupby("order_channel").agg({"master_id":"count",
                                 "total_order_sum":"sum",
                                 "total_value_sum":"sum"})

          
df["total_value_sum"].sort_values(ascending=False).head(10)  # bu yanlış oldu
           
df.sort_values("total_order_sum",ascending=False).head(10)   # bu doğru
           


df["last_order_date"].max()
today_date=dt.datetime(2021,6,1)

type(today_date)

rfm=pd.DataFrame()
rfm["customer_id"]=df["master_id"]
rfm["recency"]=today_date-df['last_order_date']
rfm["frequency"]=df["total_order_sum"]
rfm["monetary"]=df["total_value_sum"]
#rfm.columns=["monetary","frequency","recency"]




rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))



seg_map={
r'[1-2][1-2]': 'hibernating',
r'[1-2][3-4]': "at_Risk",
r'[1-2]5': "cant_loose",
r'3[1-2]': 'about_to_sleep',
r'33': 'need_attention',
r'[3-4][4-5]': 'loyal_customers',
r'41': 'promising',
r'51': 'new_customers',
r'[4-5][2-3]': 'potential_loyalists',
r'5[4-5]': "champions"
}

rfm["segment"]=rfm["RFM_SCORE"].replace(seg_map,regex=True)

rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg({"mean","count"})




rfm.groupby("segment").agg({"recency":"mean",
                                       "frequency":"mean",
                                       "monetary":"mean"})

          

rfm["interested_in_categories_12"]=df["interested_in_categories_12"]


rfm[(rfm["segment"].isin(["champions", "loyal_customers"]))&(rfm["interested_in_categories_12"].apply(lambda x: "KADIN" in x))&(rfm["monetary"]>250)]["customer_id"].to_csv("yeni_marka_hedef_müşteri_id.cvs",index=False)


  


# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
