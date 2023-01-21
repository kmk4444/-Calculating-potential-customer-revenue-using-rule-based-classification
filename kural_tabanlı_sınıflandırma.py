##### GÖREV 1 ######

#Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df=pd.read_csv("WEEK_2/Ödevler/persona.csv")
df.info()
df.head()
# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
{col: df[col].nunique() for col in df.columns}
df.nunique()

# Soru 3: Kaç unique PRICE vardır?
df[["PRICE"]].nunique()

#§ Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df.groupby("PRICE")["PRICE"].count()

# § Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df.head()
df.groupby("COUNTRY")["PRICE"].count()

#§ Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY")["PRICE"].sum()

#§ Soru 7: SOURCE türlerine göre satış sayıları nedir?

df.groupby("SOURCE")["PRICE"].count()
df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY")["PRICE"].mean()

# § Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.pivot_table("PRICE","COUNTRY","SOURCE", aggfunc="mean")

######## GÖREV 2 ########
# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

variables = ["COUNTRY","SOURCE","SEX","AGE"]
agg_df = df.groupby(variables).agg({"PRICE": "mean"})
print(agg_df)

##### GÖREV 3 ######

#• Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
#• Çıktıyı agg_df olarak kaydediniz.

agg_df = agg_df.sort_values("PRICE", ascending = False)


# Görev 4:  Indekste yer alan isimleri değişken ismine çeviriniz.

agg_df = agg_df.reset_index()
agg_df.head()


# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’eekleyiniz
from matplotlib import pyplot as plt
df["AGE"].hist()
plt.show()

agg_df['AGE_CAT'] = pd.cut(x = agg_df['AGE'], bins = [0,18,23,30,40,70], labels=['0_18', '19_23', '24_30', '31_40', '41_70'])
agg_df.head()
agg_df.info()


# Görev 6:  Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

out_age_price=agg_df.drop(["AGE", "PRICE"], axis=1)
agg_df["CUSTOMERS_LEVEL_BASED"]=["_".join(i).upper() for i in out_age_price.values]
agg_df = agg_df.groupby("CUSTOMERS_LEVEL_BASED").aggregate({"PRICE":"mean"})
agg_df.head()

#2.yol upper yapmak için
agg_df["CUSTOMERS_LEVEL_BASED"]=["_".join(i) for i in out_age_price.values]
agg_df = agg_df.groupby("CUSTOMERS_LEVEL_BASED").aggregate({"PRICE":"mean"})
agg_df.head()
agg_df = agg_df.reset_index() # groupbydan sonra index yapıyor, upper yapamıyorum.
agg_df.info()
agg_df["CUSTOMERS_LEVEL_BASED"] = agg_df["CUSTOMERS_LEVEL_BASED"].str.upper()
agg_df.index = agg_df["CUSTOMERS_LEVEL_BASED"]
agg_df = agg_df.drop("CUSTOMERS_LEVEL_BASED", axis=1)
agg_df.head()

#Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels = ["D", "C", "B", "A"])
agg_df.head()
agg_df.info()

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]})

# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.

agg_df = agg_df.reset_index()
agg_df.head()
def predict(df,country,source,sex,age_cat):
    new_user = str(country) + "_" + str(source) + "_" + str(sex) + "_" + str(age_cat)
    print(df[df["CUSTOMERS_LEVEL_BASED"]== new_user])

predict(agg_df,"TUR", "ANDROID", "FEMALE", "31_40")
predict(agg_df,"FRA", "IOS", "FEMALE", "31_40")
