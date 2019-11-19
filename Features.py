import pandas as pd
import numpy as np 
import missingpy
from sklearn.linear_model import Lasso

train = pd.read_csv('train.csv')
macro = pd.read_csv('macro.csv')

train.isna().sum().sort_values()[0:50]
features_train = ["full_sq","metro_min_walk","big_market_km",
                  "workplaces_km","university_km","cafe_count_1000","shopping_centers_km","office_km",
                  "big_church_km","school_education_centers_top_20_raion","build_count_after_1995",
                  "cafe_count_1500_price_500","market_count_500","oil_chemistry_km","railroad_km","ts_km",
                  "young_all","work_male","work_female","ekder_all","build_count_mix",
                  "build_count_1971-1995","build_count_1946-1970","build_count_1921-1945","build_count_before_1920"]

features_subset = train[features_train]
#features_subset[features_subset["build_count_mix"].isna()][["build_count_mix","build_count_1971-1995","build_count_1946-1970"]].head()

features_subset["metro_min_walk"] = features_subset["metro_min_walk"].fillna(features_subset["metro_min_walk"].mean())
rf_imp = missingpy.MissForest(criterion = "mse")
filled_features = rf_imp.fit_transform(features_subset)
data_imp = pd.DataFrame(filled_features, columns = features_subset.columns)
data_imp["target"] = train["price_doc"]
data_imp["timestamp"] = train["timestamp"]
l_feat = Lasso(alpha = .4)
l_feat.fit(data_imp.drop(["timestamp","target"], axis = 1),data_imp["target"])
coefs = l_feat.coef_

selected_features = []
cols = data_imp.drop(["timestamp","target"], axis = 1).columns
for i in range(len(coefs)):
    if coefs[i] != 0:
        selected_features.append(cols[i])
#selected_features

subset_feature = data_imp[selected_features]
subset_feature["target"] = data_imp["target"]
subset_feature["timestamp"] = data_imp["timestamp"]
macro.drop(index = 2484, inplace = True)

macro["timestamp"] = pd.to_datetime(macro["timestamp"])

macro_train = ["timestamp","unemployment","salary","construction_value","rent_price_1room_eco","rent_price_1room_bus",
               "cpi","childbirth","marriages_per_1000_cap","eurrub","deposits_value"]
macro_subset = macro[macro_train]

macro_subset.set_index("timestamp", inplace = True)
macro_filled = macro_subset.interpolate(method = "time")
macro_filled.fillna(method = "bfill", inplace = True)
macro_filled.reset_index(inplace = True)
subset_feature["timestamp"] = pd.to_datetime(subset_feature["timestamp"])
merged = subset_feature.merge(macro_filled, on='timestamp', how='left')
merged.to_csv("final_NN_feat.csv")
