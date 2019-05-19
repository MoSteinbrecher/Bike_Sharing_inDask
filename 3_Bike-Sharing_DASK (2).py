#!/usr/bin/env python
# coding: utf-8

# # Statistical Programming with Python: Bike Sharing Prediction
# ## Group Assignment
# By Team O2-2 (B)

# ### 1. Exploratory Data Analysis (descriptive analytics)

# In[311]:


import numpy as np
from distributed import Client, progress
import dask
import dask.dataframe as dd
import dask.array as da
import dask_ml
from dask.distributed import Client, progress
from dask_ml.preprocessing import Categorizer, DummyEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from dask_ml.preprocessing import DummyEncoder
from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import mean_squared_error
from dask_ml.xgboost import XGBRegressor
import dask.array as da
from dask_ml.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error  
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import explained_variance_score
import seaborn as sns
import missingno as msno
from pandas.api.types import is_datetime64tz_dtype
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import kaggle
import zipfile


# In[70]:


Client()


# In[295]:


client = Client()
client


# In[22]:


# Reading data through Kaggle API
kaggle.api.authenticate()
hour_df = dd.read_csv("https://gist.githubusercontent.com/geraldwal/b5a83f4c670abe0a662abce558e5d433/raw/bce4bbfc63355606e4503964e25798b5d2190b9b/hour%2520-%2520Python%2520Bike%2520Sharing",sep=",",
    parse_dates=["dteday"])


# #### Data quality

# First we will take a look at the quality of the variables and the possible relationship between them.

# In[23]:


hour_df


# In[24]:


hour_df.dtypes


# Not necessary right now
# Categorizing Variables for Dummy Encoding.

# In[34]:


hour_df.categorize


# We will try to predict the count value as it is the sum of the casual and registered. This means we have to leave these values out for the modelling.

# In[36]:


# Predict cnt so casual and registered can be left out
hour_df = hour_df.drop(["casual", "registered"], axis=1)


# Initially, we tried to encode the "year" variable as 2011 and 2012. This, however, does not add any informational value for a categorical variable and as it decreased our R2 scores from our predictions, we removed this. See original code below

# In[6]:


# Set the right values for yr
#hour_df.yr = hour_df.yr.replace({0: 2011, 1: 2012})


# To check for missing values in our dataset, we used the seaborn library to build a heatmap, indicating potential missing values in the dataset

# In[39]:


# Check for missing values
msno.matrix(hour_df.compute(), figsize=(12, 5))


# Another problem would be the possible outliers in the dataset.
# We checked the outliers using a boxplot representation

# In[58]:


# Check for outliers using boxplots
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=hour_df.compute(), y="cnt", orient="v", ax=axes[0][0])
sns.boxplot(data=hour_df.compute(), y="cnt", x="season", orient="v", ax=axes[0][1])
sns.boxplot(data=hour_df.compute(), y="cnt", x="hr", orient="v", ax=axes[1][0])
sns.boxplot(data=hour_df.compute(), y="cnt", x="workingday", orient="v", ax=axes[1][1])

axes[0][0].set(ylabel="Cnt", title="Box Plot On Count")
axes[0][1].set(xlabel="Season", ylabel="Cnt", title="Box Plot On Count Across Season")
axes[1][0].set(
    xlabel="Hour Of The Day",
    ylabel="Cnt",
    title="Box Plot On Count Across Hour Of The Day",
)
axes[1][1].set(
    xlabel="Working Day", ylabel="Cnt", title="Box Plot On Count Across Working Day"
)


# As seen in the outlier detection, there where some outliers located in the numerical variables. We therefore tested our predictions with removing outliers outside three standard deviations from the mean of each variable. This process yielded the highest results when removing outliers only for "humidity"

# In[60]:


# Eliminate the outliers detected using the boxplot.
#hour_no = hour_df[
    #np.abs(hour_df["windspeed"] - hour_df["windspeed"].mean()) <= (3 * hour_df["windspeed"].std())
#]

#hour_no = hour_no[
    #np.abs(hour_no["temp"] - hour_no["temp"].mean()) <= (3 * hour_no["temp"].std())
#]

hour_no = hour_df.compute()[
    np.abs(hour_df.compute()["hum"] - hour_df.compute()["hum"].mean()) <= (3 * hour_df.compute()["hum"].std())
]


# #### Data visualization

# Now that we ensured the data quality regarding outliers and msising values; the next step is to look for correlations between different variables in the dataset using a correlation matrix. There is an almost 1 to 1 relationship between temp and atemp which means we will leave one of them out for our modelling. In this case temp has the biggest effect on the count value so we leave out atemp.

# In[307]:


# Check the correlations of the different variables.
corrMatt = hour_no[
    [
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "workingday",
        "weekday",
        "season",
        "holiday",
        "hr",
        "cnt",
    ]
].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
sns.heatmap(corrMatt, mask=mask, vmax=0.8, square=True, annot=True)


# In[62]:


# Removing temp because of the almost 1 to 1 correlation with atemp.
hour_no = hour_no.drop(["atemp"], axis=1)


# Visualizing the data will show important insights for the modelling phase.
# First of all we will take a look at the variables that showed the highest correlation with the cnt values in the correlation matrix: temp and hum. Showing a clearly linear relation between these two variable and the count variable.

# In[63]:


# Plot the temperature against the usage.
fig, ax1 = plt.subplots()
fig.set_size_inches(15, 5)
sns.pointplot(x="temp", y="cnt", data=hour_no.compute(), join=True, ax=ax1)
ax1.set(
    xlabel="Temperature",
    ylabel="Users Count",
    title="Average Users Per Temperature",
    label="big",
)


# In[65]:


# Plot the humidity against the usage.
fig, ax6 = plt.subplots()
fig.set_size_inches(15, 5)
sns.pointplot(x="hum", y="cnt", data=hour_no.get(), join=True, ax=ax6)
ax6.set(
    xlabel="Humidity",
    ylabel="Users Count",
    title="Average Users Per Humidity",
    label="big",
)


# Now it will be interesting to take a look at the distribution of the rentals over the different months.

# In[47]:


# Plot Monthly Distribution
fig, ax1 = plt.subplots()
fig.set_size_inches(15, 5)

sns.barplot(data=hour_no, x="mnth", y="cnt", ax=ax1)
ax1.set(xlabel="Month", ylabel="Average Count", title="Average Count By Month")


# Looking at the distribution of the rentals on a daily basis there are clearly periods of higher demand during a day, ie the moments people have to get home from/ go to work/school, and the other moments of the day.
# Of course there is a big seasonal effect as well as an effect of wheteher it is a working day or not.

# In[69]:


# Plot hourly distributions regarding season, day of week, workday
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
fig.set_size_inches(15, 18)

hourAggregated = dd.DataFrame(
    hour_no.groupby(["hr", "season"], sort=True)["cnt"].mean()
).reset_index()
sns.pointplot(
    x=hourAggregated["hr"],
    y=hourAggregated["cnt"],
    hue=hourAggregated["season"],
    data=hourAggregated,
    join=True,
    ax=ax1,
)
ax2.set(
    xlabel="Hour Of The Day",
    ylabel="Users Count",
    title="Average Users Count By Hour Of The Day Across Season",
    label="big",
)

hourAggregated = dd.DataFrame(
    hour_no.groupby(["hr", "weekday"], sort=True)["cnt"].mean()
).reset_index()
sns.pointplot(
    x=hourAggregated["hr"],
    y=hourAggregated["cnt"],
    hue=hourAggregated["weekday"],
    data=hourAggregated,
    join=True,
    ax=ax2,
)
ax3.set(
    xlabel="Hour Of The Day",
    ylabel="Users Count",
    title="Average Users Count By Hour Of The Day Across Weekdays",
    label="big",
)

hourAggregated = dd.DataFrame(
    hour_no.groupby(["hr", "workingday"], sort=True)["cnt"].mean()
).reset_index()
sns.pointplot(
    x=hourAggregated["hr"],
    y=hourAggregated["cnt"],
    hue=hourAggregated["workingday"],
    data=hourAggregated,
    join=True,
    ax=ax3,
)
ax3.set(
    xlabel="Hour Of The Day",
    ylabel="Users Count",
    title="Average Users Count Workingday vs Weekend",
    label="big",
)


# #### Data visualization

# The windspeed variable is showing a lot of 0 values, while it is almost impossible that there is 0 wind. That is why we decided to impute the 0 values of windspeed using a random forest regressor based on season, weather situation, humidity, month, temperature and year to estimate the values of windspeed in these cases.

# In[49]:


# Check number of 0 values in windspeed
len(hour_no[hour_no["windspeed"] == 0])


# In[80]:


da.dataWind0 = hour_no[hour_no["windspeed"] == 0]
da.dataWindNot0 = hour_no[hour_no["windspeed"] != 0]
rfModel_wind = RandomForestRegressor()
da.windColumns = ["season", "weathersit", "hum", "mnth", "temp", "yr"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

da.wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])
da.dataWind0["windspeed"] = wind0Values
da.dataws = dataWindNot0.append(dataWind0)


# In[81]:


dataws.sort_index(inplace=True)
dataws.head()


# #### Feature Creation

# After transforming, imputing, deleting and creating some variables, we were able to create the final dataset (dataws) and made sure it uses the right formats for the different datatypes.

# In[82]:


# Convert final data in right format
dataws["season"] = dataws["season"].astype("category")
dataws["yr"] = dataws["yr"].astype("category")
dataws["mnth"] = dataws["mnth"].astype("category")
dataws["hr"] = dataws["hr"].astype("category")
dataws["weekday"] = dataws["weekday"].astype("category")
dataws["workingday"] = dataws["workingday"].astype("category")
dataws["holiday"] = dataws["holiday"].astype("category")
dataws["weathersit"] = dataws["weathersit"].astype("category")


# In[83]:


# Convert final data in right format
dataws["temp"] = dataws["temp"].astype("float")
dataws["hum"] = dataws["hum"].astype("float")
dataws["windspeed"] = dataws["windspeed"].astype("float")


# In[84]:


dataws


# ### 2. Machine Learning (predictive analytics)

# ### 2.1 Linear regression

# We will start using the dataset with no outlayers and without the wind values inputed using the random forest, to check later the positive or negative influence of this inputation to the model results.
# 
# We know that the we have to predict the cnt values from the first of October of 2012, so we identify this record to make our train/test split

# We will start using the dataset with no outlayers and without the wind values inputed using the random forest, to check later the positive or negative influence of this inputation to the model results.
# 
# We know that the we have to predict the cnt values from the first of October of 2012, so we identify this record to make our train/test split

# In[195]:


hour_no[hour_no["dteday"] == "2012-10-01"].head(1)
hour_no_train = hour_no[hour_no["instant"]<15212]
hour_no_train = hour_no[hour_no["instant"]>=15212]


# First we get rid of date and instance, as they do not add info to the model

# In[196]:


hour_no_to_use = hour_no_train.loc[:, "season":"windspeed"]
hour_no_to_use.shape


# In[197]:


hour_no_train = hour_no.iloc[0:15212,:]
hour_no_test = hour_no.iloc[15212:17135]


# We check the percentage of data that we are using for test

# In[198]:


hour_no_test.shape[0]/(hour_no_train.shape[0]+hour_no_test.shape[0])


# We now define the columns for the model fit. We start first without binarizing "cathegorical ones", and try latter so see the results after OneHotEncoding them. We will not use instant as it will add no information and date as we have the date info splitted into other variables

# In[199]:


hour_no_train.columns


# In[200]:


hour_no_train_X = hour_no_train.loc[:, "season":"windspeed"]


# Check the results of the dataset to train

# In[201]:


hour_no_train_X.head(1)


# In[202]:


hour_no_train_label = hour_no_train.loc[:,"cnt"]
hour_no_train_label.head()


# Now we build the most basic model with Linear Regression

# In[203]:


LR = LinearRegression(fit_intercept=True)


# In[204]:


da.LR_model_baseline = LR.fit(hour_no_train_X.values, hour_no_train_label.values)


# In[205]:


da.hour_no_test_X = hour_no_test.loc[:, "season":"windspeed"]


# In[206]:


LR_baseline_predicted = LR_model_baseline.predict(hour_no_test_X.values)


# In[207]:


da.actual_label_test = hour_no_test.loc[:, "cnt"]


# Linear Regression Model definiton 

# In[113]:


#def LinReg(hour_no_train_X, hour_no_test_X, hour_no_train_label, actual_label_test):
    lr = LinearRegression()
    lr.fit(hour_no_train_X, hour_no_train_label)
    y_pred = lm.predict(hour_no_test_X)
    print("Intercept:", lr.intercept_)
    print("Coefficients:", lr.coef_)
    print("Mean squared error (MSE): {:.2f}".format(mean_squared_error(actual_label_test, y_pred)))
    print(
        "Variance score (R2): {:.2f}".format(
            r2_score(actual_label_test.compute(), y_pred.compute())
        )
    )
    return y_pred


# In[116]:


#LR_baseline_predicted = LinReg.predict(hour_no_test_X)


# In[208]:


print("R2:")
r2_score(
    actual_label_test,
    da.LR_baseline_predicted,
    sample_weight=None,
    multioutput="uniform_average",
)


# In[315]:


mean_squared_error(actual_label_test, LR_baseline_predicted, multioutput="uniform_average")


# We see from the R squared and MSE that the result is not good. We will encode our categorical variables to find a better result

# In[219]:


hour_no.columns


# We select the ones that are "categorical" meaning that even being numerical variables its meaning is not numeric

# In[220]:


to_bin = hour_no[['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit']]


# In[221]:


enc = OneHotEncoder(handle_unknown='ignore')


# As we will normalize our numerical variables later, we want to avoid that the OneHotEncoder gives artificially more weight than the numeric variables. For this reason we multiply by 0.7

# In[222]:


matrix_enc = enc.fit_transform(to_bin)*0.7


# In[223]:


matrix_enc_d = pd.DataFrame(matrix_enc.todense())


# We add back the values of our numeric variables. For doing this we re set the index to the original one. Otherwise we find problems when adding the variables from the old datase

# In[224]:


matrix_enc_d = matrix_enc_d.set_index(hour_no.index)


# In[225]:


matrix_enc_d["temp"] = hour_no["temp"]
matrix_enc_d["hum"] = hour_no["hum"]
matrix_enc_d["windspeed"] = hour_no["windspeed"]


# In[226]:


matrix_enc_d.shape


# Now we split again between test and train

# In[227]:


enc_train = matrix_enc_d.iloc[0:15212,:]
enc_test = matrix_enc_d.iloc[15212:17135,:]


# We make sure that the dimensions are correct

# In[228]:


len(enc_train)+len(enc_test)==len(hour_no)


# Now we are ready to use our linear model now to fit on our binarized dataset

# In[229]:


LR_model_HotEncoded = LR.fit(enc_train.values,hour_no_train["cnt"].values)


# In[230]:


predictions_encoded = LR_model_HotEncoded.predict(enc_test.values)


# In[231]:


print("R2:")
r2_score(
    actual_label_test,
    predictions_encoded,
    sample_weight=None,
    multioutput="uniform_average",
)


# In[316]:


mean_squared_error(predictions_encoded, LR_baseline_predicted, multioutput="uniform_average")


# So we see that the OneHotEncoder, as expected, improves the prediction thoroughly. Now the next step will me normalizing our numeric variables and check the results

# We take each of our numeric the variables, convert them into a numpy array in order to reshape in such a form that the MinMaxScaler can understand. Mi

# In[233]:


temp_norm = np.array(hour_no["temp"])
temp_norm_rs = temp_norm.reshape(-1, 1)
normalizer = preprocessing.MinMaxScaler()
temp_nor_out = normalizer.fit_transform(temp_norm_rs)
temp_nor_out


# We plot the results to check that it works fine

# In[234]:


fig, ax = plt.subplots()
ax.plot(temp_nor_out)

ax.set(xlabel='time (hours)', ylabel='values',
       title='temperature normalized')
ax.grid()

fig.savefig("test.png")
plt.show()


# In[235]:


hum_norm = np.array(hour_no["hum"])
hum_norm_rs = hum_norm.reshape(-1, 1)
normalizer = preprocessing.MinMaxScaler()
hum_nor_out = normalizer.fit_transform(hum_norm_rs)
hum_nor_out


# In[236]:


fig, ax = plt.subplots()
ax.plot(hum_nor_out)

ax.set(xlabel='time (hours)', ylabel='values',
       title='hum normalized')
ax.grid()

fig.savefig("test.png")
plt.show()


# In[237]:


windspeed_norm = np.array(hour_no["windspeed"])
windspeed_norm_rs = windspeed_norm.reshape(-1, 1)
normalizer = preprocessing.MinMaxScaler()
windspeed_nor_out = normalizer.fit_transform(windspeed_norm_rs)
windspeed_nor_out


# In[238]:


fig, ax = plt.subplots()
ax.plot(windspeed_nor_out)

ax.set(xlabel='time (hours)', ylabel='values',
       title='wind normalized')
ax.grid()

fig.savefig("test.png")
plt.show()


# Now we build them back together in a dataframe to put them back in our model

# In[239]:


numeric_scaled = pd.DataFrame({"temp_norm":temp_norm})
numeric_scaled["hum_norm"]=hum_nor_out
numeric_scaled["windspeed_norm"]=windspeed_nor_out


# In[240]:


numeric_scaled.head()


# In[241]:


numeric_scaled.shape


# Next step is append this columns to our OneHotEncoded dataframe with the binarized cathegories

# In[242]:


matrix_enc_d_norm = pd.DataFrame(matrix_enc.todense())


# In[243]:


matrix_enc_d_norm["temp_norm"] = numeric_scaled["temp_norm"]
matrix_enc_d_norm["hum_norm"] = numeric_scaled["hum_norm"]
matrix_enc_d_norm["windspeed_norm"] = numeric_scaled["windspeed_norm"]


# In[244]:


matrix_enc_d_norm.shape


# Train test split again

# In[245]:


enc_norm_train = matrix_enc_d_norm.iloc[0:15212,:]
enc_norm_test = matrix_enc_d_norm.iloc[15212:17135,:]


# Fit new model

# In[248]:


LR_model_Encoded_Norm = LR.fit(enc_norm_train.values,hour_no_train["cnt"].values)


# In[249]:


predictions_encoded_norm = LR_model_Encoded_Norm.predict(enc_test.values)


# In[251]:


print("R2:")
r2_score(
    actual_label_test,
    predictions_encoded_norm,
    sample_weight=None,
    multioutput="uniform_average",
)


# In[317]:


mean_squared_error(predictions_encoded_norm, LR_baseline_predicted, multioutput="uniform_average")


# The result of out scaling is actually negative for the prediction, so we will not use standarization in the next iteration

# Finally we try using the windspeed inputation and OneHotEncoding

# First we delete our winspeed column from the training dataset

# In[266]:


data_enc_windinputed = matrix_enc_d.loc[:, :"hum"]


# Now we include the imputed values for winspeed

# In[267]:


data_enc_windinputed["wind_inputed"] = dataws["windspeed"]


# In[268]:


data_enc_windinputed.head()


# In[269]:


enc_ws_inputed_train = data_enc_windinputed.iloc[0:15212,:]
enc_ws_inputed_test = data_enc_windinputed.iloc[15212:17135,:]


# In[270]:


LR_model_Encoded_ws_inputed = LR.fit(enc_ws_inputed_train.values,hour_no_train["cnt"].values)


# In[271]:


predictions_encoded_ws_inputed = LR_model_Encoded_ws_inputed.predict(enc_test.values)


# In[272]:


print("R2:")
r2_score(
    actual_label_test,
    predictions_encoded_ws_inputed,
    sample_weight=None,
    multioutput="uniform_average",
)


# So we see that using the inputation oif wind does not give us a better result in terms of R2

# ### 2.2 Random forest

# Our baseline model did not yield high results, more complex models could solve this issue. In our first attempt to improve our R2 score we will also model with a random forest algorithm, this is done because our dataset contains both categorical and numerical variables

# We used the hotencoded variables from our linear regression to perform our random forest, this dataset is just using the hotencoded ones, before normalizing

# In[283]:


enc_train = matrix_enc_d.iloc[0:15212,:]
enc_test = matrix_enc_d.iloc[15212:17135,:]
yLabels = hour_no_train["cnt"]


# In[285]:


regressor = RandomForestRegressor()
parameters = [
    {"n_estimators": [100, 150, 200, 250, 300], "max_features": ["auto", "sqrt", "log2"]}
]
grid_search = GridSearchCV(estimator=regressor, param_grid=parameters)
grid_search = grid_search.fit(enc_train, yLabels)
parameters_to_use = grid_search.best_params_


# In[286]:


parameters_to_use


# In[287]:


rfModel = RandomForestRegressor(n_estimators=200)
rfModel.fit(enc_train,hour_no_train["cnt"])
preds_1st_forest = rfModel.predict(X= enc_test)


# In[288]:


r2_score(y_pred=preds_1st_forest, y_true=hour_no_test["cnt"])


# The random forrest model is performing better than the linear regression but still leaves room for improvement. Lets first check how our model performs when we use the encoded variables including the predicted missing variables we used earlier. We also ran this model using the normalized encoded variables, however, this did not yield better results, most likely because of skewness in the original dataset

# First, lets call the wind imputed variables

# In[289]:


enc_ws_inputed_train = data_enc_windinputed.iloc[0:15212,:]
enc_ws_inputed_test = data_enc_windinputed.iloc[15212:17135,:]
yLabels = hour_no_train["cnt"]


# In[290]:


regressor = RandomForestRegressor()
parameters = [
    {"n_estimators": [100, 150, 200, 250, 300], "max_features": ["auto", "sqrt", "log2"]}
]
grid_search = GridSearchCV(estimator=regressor, param_grid=parameters)
grid_search = grid_search.fit(enc_ws_inputed_train, yLabels)
parameters_to_use = grid_search.best_params_


# In[291]:


parameters_to_use


# In[292]:


rfModel = RandomForestRegressor(n_estimators=200)
rfModel.fit(enc_train,hour_no_train["cnt"])
preds_2nd_forest = rfModel.predict(X= enc_test)


# In[293]:


r2_score(y_pred=preds_2nd_forest, y_true=hour_no_test["cnt"])


# This model is more or less similar to our results from the first random forest. Lets therefore see if we might not be overfitting our data when using only randomforests. 

# ### 2.3 XB Boost

# The XGboost model works in a similar fashion as the gradient boost, but is considered faster and mitigates better 

# In[297]:


enc_train = matrix_enc_d.iloc[0:15212,:]
enc_test = matrix_enc_d.iloc[15212:17135,:]
yLabels = hour_no_train["cnt"]


# In[298]:


xg_reg = xgb.XGBRegressor(max_depth = 5, alpha = 10, n_estimators = 5000, colsample_bytree=0.3)


# In[299]:


xg_reg.fit(enc_train,yLabels)


# In[300]:


xgpreds = xg_reg.predict(enc_test)


# In[301]:


len(xgpreds)


# In[302]:


print ("R2 Value For Gradient Boost Regression: ",r2_score(y_pred=xgpreds, y_true=hour_no_test["cnt"]))


# ### 2.4 Conclusion

# As illustrated below, our target variable is best predicted using an XGBoost model due to the high number of features we encoded and the multicoliniarity between 

# In[308]:


y_test=hour_no_test["cnt"]


# In[309]:


LG_r2 = metrics.r2_score(
    actual_label_test,
    predictions_encoded_norm,
    sample_weight=None,
    multioutput="uniform_average",
)
RF_r2 = float(r2_score(y_pred=preds_1st_forest, y_true=y_test))
Rid_r2 = float(r2_score(y_pred=ridge_preds, y_true=y_test))
XGB_r2 = float(r2_score(y_pred=xgpreds, y_true=y_test))
GrB_r2 = float(r2_score(y_pred=preds_boost, y_true=y_test))
data = [
    {
        "LogRegression": LG_r2,
        "RandomForest": RF_r2,
        "Ridge": Rid_r2,
        "XGBoost": XGB_r2,
        "GrB": GrB_r2,
    },
]

comparisonmetrics = pd.DataFrame(
    data,
    index=["r2"],
    columns=[
        "LogRegression",
        "RandomForest",
        "Ridge",
        "XGBoost",
        "GrB",
    ],
)

comparisonmetrics.round(2)


# In[310]:


# here we plot the predictions of the various models against the actual values, so to compare them again.
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(30, 25))

ax1 = axs[0, 0]
ax1.plot(y_test, y_test, "r--", y_test, predictions_encoded_norm, "b,")
ax1.set_title("LogReg")

ax2 = axs[0, 1]
ax2.plot(y_test, y_test, "r--", y_test, preds_1st_forest, "b,")
ax2.set_title("RandomForest")

ax1 = axs[1, 0]
ax1.plot(y_test, y_test, "r--", y_test, ridge_preds, "b,")
ax1.set_title("Ridge")

ax2 = axs[1, 1]
ax2.plot(y_test, y_test, "r--", y_test, xgpreds, "b,")
ax2.set_title("GrdB")

ax1 = axs[2, 0]
ax1.plot(y_test, y_test, "r--", y_test, preds_boost, "b,")
ax1.set_title("XGBoost")

