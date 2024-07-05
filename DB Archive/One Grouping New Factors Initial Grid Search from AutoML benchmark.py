# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM Regressor training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **13.1.x-gpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/2864270915671646).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "avgprice15"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(run_id="1bf5f6c8fcde4a4e84fb195cafed7756", artifact_path="data", dst_path=input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `["PNCDEligible"]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["transUnionConceptsGC", "frthOccupationType", "transUnionConceptsWD", "ADFreqTheoOccGroupP2", "transUnionConceptsJWB", "mainDriverAgeMonths", "transUnionConceptsOOB", "transUnionConceptsDB", "transUnionConceptsW", "vehicleValue", "transUnionConceptsBF", "coveaRatingArea", "ADFreqVG2022", "piOccupationType", "transUnionConceptsTF", "THFreqVGEXTRA", "totVandalismClaims", "THSevVGEXTRA", "totFaultClaimsInLast5years", "THFreqRA2022", "totFireClaims", "THSevRA2022", "ADSevVG2022", "transUnionConceptsCSB", "transUnionConceptsMSB", "ncdprotected", "FRFreqVG2022", "ADSevVGEXTRA", "mainEmployerBusinesstype", "mainemploymentType", "catConverterFlag", "mainDrivingExperience", "WSSevVG2022", "adOccupationType", "SMFreqRA2022", "mainOccupationType", "distFromAccManCo", "vehicleKeeper", "transUnionConceptsYCB", "vehicleAEBType", "addDrivingExperience", "THSevVG2022", "largeOccupationType", "transUnionConceptsQUB", "transUnionConceptsXSB", "PIRatingArea", "ADRatingArea", "transUnionConceptsKCB", "transUnionConceptsETB", "latestConviction", "transUnionConceptsMDB", "totTheftClaims", "PDSevRA2022", "WSRatingArea", "SMSevRA2022", "pdOccupationType", "THFreqVG2022", "PISevRA2022", "ADFreqRA2022", "ncdallowed_capped", "totConvictionsInLast5years", "mainDrivingExperienceMonths", "quote_lag_calc", "FRTHSmRatingArea", "SMFreqVG2022", "classofuse", "largeRatingArea", "transUnionConceptsTRB", "ageDifference", "totWindScreenClaims", "thfreqtheooccgroup", "garaged", "registrationYear", "latestNonFaultAccClaims", "addAccessToOtherVehs", "PDRatingArea", "proposerHomeOwner", "distFromAccManCoFullPC", "totFaultAccclaims", "PDSevVG2022", "transUnionConceptsJF", "PDFreqRA2022", "latestWindscreenClaims", "transUnionConceptsLDB", "vehicleManufacturer", "vehicleAgeAtPurchase", "WSFreqRA2022", "latestFaultAccClaims", "totNonFaultAccclaims", "transUnionConceptsAD", "smfreqtheooccGroup", "vehicleAge", "PDPIPropRA2022", "wsfreqtheooccGroup", "mainmaritalstatus", "XSPIRatingArea", "FRSevVG2022", "transUnionConceptsNF", "transUnionConceptsGSB", "vehicleModel", "voluntaryexcess", "tvRegion", "mainLicenceType", "transUnionConceptsITB", "othersvehs_FE", "ADFreqVGEXTRA", "pdrcode", "vehicleFuelType", "ADSevRA2022", "PDFreqVG2022", "transUnionConceptsNE", "transUnionConceptsODB", "vehicleMarketingMultiplier", "secondcarflag", "yearsOwned", "PDPIPropVG2022", "wsOccupationType", "mostSevereConviction", "mainDriverAge", "vehicleXSPIMultiplier", "addOtherVehsOwned", "vehicleOwner", "transUnionConceptsEF", "ADFreqTheoOccGroupP1", "maindriverexperience_uk", "mosaicType", "addDriverAge", "transUnionConceptsYSB", "transUnionConceptsOE", "perc_drv_uk", "WSFreqVG2022", "cover", "vehicleBodyType", "annualMileage", "transUnionConceptsXTB", "WSSevRA2022", "pdfreqtheooccGroup", "acorn"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boolean columns
# MAGIC For each column, impute missing values and then convert into ones and zeros.

# COMMAND ----------

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(#missing_values=-999, #No sent values for missings
                                     strategy="median",
                                     ))#,("scaler", StandardScaler()) #No scaling needed
           ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', 
                                   unknown_value=-1
                                   ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="object")),
        ("cat", categorical_transformer, selector(dtype_include="object")),
    ]
    , remainder="passthrough"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC
# MAGIC `_automl_split_col_0000` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.

# COMMAND ----------

# AutoML completed train - validation - test split internally and used _automl_split_col_0000 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "val"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "test"]

# Separate target column from features and drop _automl_split_col_0000
X_train = split_train_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_test = split_test_df[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/2864270915671646)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

import lightgbm
from lightgbm import LGBMRegressor

#help(LGBMRegressor)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

from mlflow.models import make_metric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

def gini_table(eval_df):
    
    ''' returns a dataframe from which a gini coefficient can be calculated
        also can create cumulative gains curves
        pred = predicted values (output from emblem) argument required is the name of the column in df
        act = actual values (number of claims) argument required is the name of the column in df
    
        3 useful outputs
        Perc of Obs and Perc of Claims can be used to create Cumulative Gains Curves
        Gini_Area can be used to calculate the gini coefficient. Each Gini_Area is the approximate area under Cumulative
        gains curve. Feel free to change to trapezium rule in future. '''
    df = eval_df[["prediction", "target"]].sort_values(by="prediction", ascending=False)
    df = df.reset_index()
    df['Cumulative Claims'] = df["target"].cumsum()
    df['Perc of Obs'] = (df.index + 1) / df.shape[0]
    df['Perc of Claims'] = df['Cumulative Claims'] / df.iloc[-1]['Cumulative Claims']
    df['gini_area'] = df['Perc of Claims'] / df.shape[0]
    return df

def calc_gini(eval_df, _builtin_metrics):
    
    ''' uses output from gini_table to calculate a gini coefficient. Formula comes from R:\Pricing\Personal Lines Pricing - Motor\Technical\21. Provident\
        4. SAS Processes\Technical MI Tools\Gini_Coefficients_and_U_Statistics\1.Motivation - GiniCoefficientpaper.pdf
        model = column name of modelled values you wish to calculate gini coefficient of.
        obs = column name of actual values (number of claims) '''
    
    d1 = gini_table(eval_df)
    Gini_coef = round((d1.sum()['gini_area'] - 0.5) *2,6)
    return(Gini_coef)

def calc_auc(eval_df, _builtin_metrics):
    '''
    uses outputs from gini_table but returns auc '''
    df = eval_df[["prediction", "target"]].sort_values(by="prediction", ascending=False)
    df = df.reset_index()
    df['Cumulative Claims'] = df["target"].cumsum()
    df['Perc of Obs'] = (df.index + 1) / df.shape[0]
    df['Perc of Claims'] = df['Cumulative Claims'] / df.iloc[-1]['Cumulative Claims']
    df['gini_area'] = df['Perc of Claims'] / df.shape[0]
    AUC = round(df.sum()['gini_area'],6)
    return(AUC)

# COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials


# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

mlflow.sklearn.autolog(disable=True)
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

gini_metric = mlflow.models.make_metric(
    eval_fn=calc_gini,
    greater_is_better=True)

def objective(params):
  with mlflow.start_run() as mlflow_run:
    lgbmr_regressor = LGBMRegressor(**params)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("regressor", lgbmr_regressor),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True,
    )
 
    model.fit(X_train, y_train, regressor__callbacks=[lightgbm.early_stopping(10), lightgbm.log_evaluation(0)], regressor__eval_set=[(X_val_processed,y_val)])

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(target_col):y_train}),
        targets=target_col,
        model_type="regressor",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_"},
        custom_metrics = [gini_metric]
    )
    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(target_col):y_val}),
        targets=target_col,
        model_type="regressor",
        evaluator_config= {"log_model_explainability": False,
                           "metric_prefix": "val_"},
        custom_metrics = [gini_metric]
   )
    lgbmr_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(target_col):y_test}),
        targets=target_col,
        model_type="regressor",
        evaluator_config= {"log_model_explainability": False,
                           "metric_prefix": "test_"},
        custom_metrics = [gini_metric]
   )
    lgbmr_test_metrics = test_eval_result.metrics

    loss = 1-lgbmr_val_metrics["val_calc_gini"]

    # Truncate metric key names so they can be displayed together
    lgbmr_val_metrics = {k.replace("val_", ""): v for k, v in lgbmr_val_metrics.items()}
    lgbmr_test_metrics = {k.replace("test_", ""): v for k, v in lgbmr_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": lgbmr_val_metrics,
      "test_metrics": lgbmr_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree regressor, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

#Lambda seems really high?
space = {
  "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.6),
  "lambda_l1": hp.uniform("lambda_l1", 0, 10),
  "lambda_l2": hp.uniform("lambda_l2", 0, 20),
  "learning_rate": 0.15, #higher learning rate to let the tuning go quicker
  "max_bin": hp.choice('max_bin', np.arange(20, 150, dtype=int)),
  "max_depth": hp.choice('max_depth', np.arange(8, 14, dtype=int)),
  "min_child_samples": hp.choice('min_child_samples', np.arange(75, 125, dtype=int)),
  "n_estimators": hp.choice('n_estimators', np.arange(2500, 5000, dtype=int)),
  "num_leaves": hp.choice('num_leaves', np.arange(250, 500, dtype=int)),
  "subsample": hp.uniform("subsample", 0.6, 0.8),
  "random_state": 566182064,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

from hyperopt import SparkTrials
trials = SparkTrials()

fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=10,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

def gini_table2(eval_df):
    
    ''' returns a dataframe from which a gini coefficient can be calculated
        also can create cumulative gains curves
        pred = predicted values (output from emblem) argument required is the name of the column in df
        act = actual values (number of claims) argument required is the name of the column in df
    
        3 useful outputs
        Perc of Obs and Perc of Claims can be used to create Cumulative Gains Curves
        Gini_Area can be used to calculate the gini coefficient. Each Gini_Area is the approximate area under Cumulative
        gains curve. Feel free to change to trapezium rule in future. '''
    df = eval_df[["prediction", "target"]].sort_values(by="prediction", ascending=False)
    df = df.reset_index()
    df['Cumulative Claims'] = df["target"].cumsum()
    df['Perc of Obs'] = (df.index + 1) / df.shape[0]
    df['Perc of Claims'] = df['Cumulative Claims'] / df.iloc[-1]['Cumulative Claims']
    df['gini_area'] = df['Perc of Claims'] / df.shape[0]
    return df

def calc_gini2(eval_df):
    
    ''' uses output from gini_table to calculate a gini coefficient. Formula comes from R:\Pricing\Personal Lines Pricing - Motor\Technical\21. Provident\
        4. SAS Processes\Technical MI Tools\Gini_Coefficients_and_U_Statistics\1.Motivation - GiniCoefficientpaper.pdf
        model = column name of modelled values you wish to calculate gini coefficient of.
        obs = column name of actual values (number of claims) '''
    
    d1 = gini_table2(eval_df)
    Gini_coef = round((d1.sum()['gini_area'] - 0.5) *2,6)
    return(Gini_coef)

def calc_auc2(eval_df):
    '''
    uses outputs from gini_table but returns auc '''
    df = eval_df[["prediction", "target"]].sort_values(by="prediction", ascending=False)
    df = df.reset_index()
    df['Cumulative Claims'] = df["target"].cumsum()
    df['Perc of Obs'] = (df.index + 1) / df.shape[0]
    df['Perc of Claims'] = df['Cumulative Claims'] / df.iloc[-1]['Cumulative Claims']
    df['gini_area'] = df['Perc of Claims'] / df.shape[0]
    AUC = round(df.sum()['gini_area'],6)
    return(AUC)

df=pd.DataFrame()
df['prediction']=model.predict(X_test)
df['target']=y_test.reset_index(drop=True)

df_perfect=pd.DataFrame()
df_perfect['prediction']=y_test
df_perfect['target']=y_test

#gini norm
print(f"Actual Gini: {calc_gini2(df)}, Perfect Gini: {calc_gini2(df_perfect)}, Gini Norm: {calc_gini2(df)/calc_gini2(df_perfect)}")


# COMMAND ----------

fi_df=pd.DataFrame()
fi_df['Feature_Name']=model["preprocessor"].get_feature_names_out()
fi_df['Feature_Gain_Values']=model['regressor'].booster_.feature_importance(importance_type='gain') #Can change to 'split' for number of splits
#fi_df.set_index('Feature_Name', inplace=True)
fi_df.sort_values('Feature_Gain_Values', ascending=False, inplace=True)
#fi_df.tail(20).plot(kind='barh')
display(fi_df)