# libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize

cars_data = pd.read_csv('autos.csv', encoding = "cp1252")

# observasi
cars_data.head(10)
cars_data.describe(include='all')
persentase_null_values = cars_data.isnull().sum()/len(cars_data)*100
persentase_null_values

# ubah format
cars_data.rename(columns={"dateCreated": "ad_created",
                          "dateCrawled": "date_crawled",
                          "fuelType": "fuel_type",
                          "lastSeen": "last_seen",
                          "monthOfRegistration": "registration_month",
                          "notRepairedDamage": "unrepaired_damage",
                          "nrOfPictures": "num_of_pictures",
                          "offerType": "offer_type",
                          "postalCode": "postal_code",
                          "powerPS": "power_ps",
                          "vehicleType": "vehicle_type",
                          "yearOfRegistration": "registration_year"},inplace=True)


# ubah tipe data waktu
cars_data['ad_created'] = pd.to_datetime(cars_data['ad_created'], format='%Y-%m-%d %H:%M:%S')
cars_data['date_crawled'] = pd.to_datetime(cars_data['date_crawled'], format='%Y-%m-%d %H:%M:%S')
cars_data['last_seen'] = pd.to_datetime(cars_data['last_seen'], format='%Y-%m-%d %H:%M:%S')

cars_data[["ad_created", "date_crawled", "last_seen"]].info()

# format string
cars_data['price']=cars_data['price'].str.replace("$","").str.replace(",","")
cars_data['price']=cars_data['price'].astype(dtype='int64')
cars_data['odometer']=cars_data['odometer'].str.replace("km","").str.replace(",","")
cars_data['odometer']=cars_data['odometer'].astype(dtype='int64')


for kolom in cars_data.columns:
    if cars_data[kolom].dtypes == 'object':
        print("kolom",kolom)
        print(cars_data[kolom].value_counts())
        print("")
        
for kolom_num in cars_data.columns:
    if cars_data[kolom_num].dtypes == 'int64':
        print("kolom",kolom_num)
        print(cars_data[kolom_num].value_counts())
        print("")

cars_data = cars_data[(cars_data['price'] >= 500) & (cars_data['price'] <=40000)]

for kolom in cars_data:
    if cars_data[kolom].dtypes == 'object':
        cars_data[kolom]=cars_data[kolom].fillna(cars_data[kolom].mode()[0])
    else:
        cars_data[kolom]=cars_data[kolom].fillna(cars_data[kolom].median())

# normalize
num_cols = cars_data._get_numeric_data().columns
cars_data[num_cols]= normalize(X=cars_data[num_cols], norm="l2", axis=1)

# encoding categorical data
le1 = preprocessing.LabelEncoder()
cars_data['abtest'] =le1.fit_transform(cars_data['abtest'])

le2 = preprocessing.LabelEncoder()
cars_data['vehicle_type'] =le2.fit_transform(cars_data['vehicle_type'])

le3 = preprocessing.LabelEncoder()
cars_data['gearbox'] =le3.fit_transform(cars_data['gearbox'])

le4 = preprocessing.LabelEncoder()
cars_data['model'] =le4.fit_transform(cars_data['model'])

le5 = preprocessing.LabelEncoder()
cars_data['fuel_type'] =le5.fit_transform(cars_data['fuel_type'])

le6 = preprocessing.LabelEncoder()
cars_data['brand'] =le6.fit_transform(cars_data['brand'])

le7 = preprocessing.LabelEncoder()
cars_data['unrepaired_damage']=le7.fit_transform(cars_data['unrepaired_damage'])