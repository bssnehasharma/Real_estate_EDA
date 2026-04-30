import streamlit as st
import pandas as pd
import numpy as np

st.title("Real Estate Data Cleaning")

if st.button("Clear Cache & Reload"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data
def load_data():
    df = pd.read_csv("india_housing_prices.parquet")
    return df

df = load_data()
st.dataframe(df.head())

st.subheader("Data Cleaning Steps")

# 1. DUPLICATES
st.write("Exact duplicates:", df.duplicated().sum())
df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
st.write("Rows after removing ID duplicates:", len(df))

# 2. MISSING VALUES
st.write("Missing before cleaning:")
st.write(df.isnull().sum()[df.isnull().sum() > 0])

# Drop rows where target is missing
df.dropna(subset=['Price_in_Lakhs'], inplace=True)

# Fill numeric cols
df['Size_in_SqFt'] = pd.to_numeric(df['Size_in_SqFt'], errors='coerce')
df['Size_in_SqFt'] = df.groupby(['City','Property_Type'])['Size_in_SqFt'].transform(
    lambda x: x.fillna(x.median())
)

st.success("Cleaning done! Final shape: {df.shape}")
st.write("Missing after cleaning:", df.isnull().sum().sum())
st.dataframe(df.head())

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🏠 Real Estate EDA Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_real_estate_data.csv")
    
    # Clean data — fix NaN issues
    df['Size_in_SqFt'] = pd.to_numeric(df['Size_in_SqFt'], errors='coerce')
    df['Price_in_Lakhs'] = pd.to_numeric(df['Price_in_Lakhs'], errors='coerce')
    df.dropna(subset=['Price_in_Lakhs', 'Size_in_SqFt'], inplace=True)
    df = df[df['Size_in_SqFt'] > 0]  
    
    # Calculate new column
    df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
    return df

df = load_data()

# ==================== 1. BASIC INFO ====================
st.header("1. Dataset Overview")
st.write("Total houses:", len(df))
st.write("Cities:", df['City'].nunique())
st.write("Average price: ₹", round(df['Price_in_Lakhs'].mean(), 1), "Lakhs")

# ==================== 2. PRICE DISTRIBUTION ====================
st.header("2. How are house prices spread?")
fig, ax = plt.subplots()
ax.hist(df['Price_in_Lakhs'].dropna(), bins=30)
ax.set_xlabel("Price in Lakhs")
ax.set_ylabel("Number of Houses")
st.pyplot(fig)
# ==================== 3. TOP 5 EXPENSIVE CITIES ====================
st.header("3. Which city is most expensive?")
city_price = df.groupby('City')['Price_per_SqFt'].mean().sort_values(ascending=False).head(5)
st.bar_chart(city_price)
# ==================== 4. SIZE VS PRICE ====================
st.header("4. Do bigger houses cost more?")
fig, ax = plt.subplots()
ax.scatter(df['Size_in_SqFt'], df['Price_in_Lakhs'], alpha=0.3)
ax.set_xlabel("Size in SqFt")
ax.set_ylabel("Price in Lakhs")
st.pyplot(fig)
# ==================== 5. BHK VS PRICE ====================
st.header("5. How does BHK affect price?")
bhk_price = df.groupby('BHK')['Price_in_Lakhs'].mean()
st.bar_chart(bhk_price)
# ==================== 6. FIND CHEAP HOUSES ====================
st.header("6. Where to invest? Cheap houses list")
city_avg = df.groupby('City')['Price_per_SqFt'].mean()
df['City_Avg'] = df['City'].map(city_avg)
cheap_houses = df[df['Price_per_SqFt'] < df['City_Avg'] * 0.9].head(10)
st.write("These 10 houses are 10% cheaper than city average:")
st.dataframe(cheap_houses[['City','Locality','BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt']])

    
   
