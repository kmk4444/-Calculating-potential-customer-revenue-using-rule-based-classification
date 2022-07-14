# Calculating Potential Customer Revenue Using Rule-Based Classification
# Business Problem:
# A game company would like to create a new level-based persona using customers' characteristics, and design segments according to the persona. '
# 'As a consequence, the company wants to estimate benefits by utilizing these segments when a new customer purchases games.

#Persona dataset has 5 variables with 5000 entries. Prıce and Age variables are numerical,
# but Source, Sex and Country variables are categorical. The dataset has no missing value.

# PRICE: Customer's spending amount
# SOURCE: The type of device the customer connects
# SEX: Customer's sex
# COUNTRY: Customer's country
# AGE: Customer's age

# Task 1: Explain questions

# Task 1.1: Read persona.csv
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df=pd.read_csv("WEEK_2/Ödevler/persona.csv")
df.info()
df.head()

# Task 1.2: describe unique values of SOURCE and find frequencies.
df["SOURCE"].value_counts()

# Task 1.3: find the number of unique values of PRICE
df["PRICE"].nunique()

# Task 1.4: find the frequencies of price variable
df["PRICE"].value_counts()

# Task 1.5: How many sales were made for each country?
df.groupby("COUNTRY")["COUNTRY"].count()

# Task 1.6: How much did countries earn from sales?
df.groupby("COUNTRY").agg({"PRICE":"sum"})

# Task 1.7: find the amount of sales by source?
df.groupby("SOURCE").agg({"SOURCE":"count"})

# Task 1.8: what is average of price by country?
df.groupby("COUNTRY").agg({"PRICE":"mean"})

# Task 1.9: what is the average of price by source?
df.groupby("SOURCE").agg({"PRICE": "mean"})

#Task 1.10: what is the average of price by country as well as source?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE":"mean"})


#Task 2: what is average price by country, source, sex and age?

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})

# Task 3: sort the values by price and save as agg_df

agg_df = (df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})).sort_values(by = "PRICE", ascending =False)
agg_df.head()

# Task 4: convert index into variables
agg_df.reset_index(inplace=True)
agg_df.head()

# Task 5: convert numeric variable into categorical variable for age
agg_df["AGE_CAT"] = pd.cut(x= agg_df["AGE"], bins = [0, 18, 23, 30, 40, 70], labels = ["0_18", "19_23", "24_30", "31_40", "41_70"])
agg_df.head()

#Task 6: create customers_level_based variable and combine all variables without price and age

out_age_price = agg_df.drop(["AGE", "PRICE"], axis=1)
agg_df["customers_level_based"] = ["_".join(i).upper() for i in out_age_price.values]
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE":"mean"})
agg_df.reset_index(inplace=True)

# Task 7: so as to describe new customers' persona, seperate four groups.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels = ["D", "C", "B", "A"])
agg_df.head()
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

# Task 8: predict customers' revenue

def ruled_based_classification(dataframe):


    def AGE_CAT(age):
        if age <= 18:
            AGE_CAT = "15_18"
            return AGE_CAT
        elif (age > 18 and age <= 23):
            AGE_CAT = "19_23"
            return AGE_CAT
        elif (age > 23 and age <= 35):
            AGE_CAT = "24_35"
            return AGE_CAT
        elif (age > 35 and age <= 45):
            AGE_CAT = "36_45"
            return AGE_CAT
        elif (age > 45 and age <= 66):
            AGE_CAT = "46_66"
            return AGE_CAT

    COUNTRY = input("Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):")
    SOURCE = input("Enter the operating system of phone (IOS/ANDROID):")
    SEX = input("Enter the gender (FEMALE/MALE):")
    AGE = int(input("Enter the age:"))
    AGE_SEG = AGE_CAT(AGE)
    new_user = COUNTRY.upper() + '_' + SOURCE.upper() + '_' + SEX.upper() + '_' + AGE_SEG

    print(new_user)
    print("Segment:" + dataframe[dataframe["customers_level_based"] == new_user].loc[:, "SEGMENT"].values[0])
    print("Price:" + str(dataframe[dataframe["customers_level_based"] == new_user].loc[:, "PRICE"].values[0]))

    return new_user

ruled_based_classification(agg_df)

