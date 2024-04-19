import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import warnings
import joblib
import joblib
from datetime import datetime as date
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.tree import export_graphviz, export_text, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from xgboost import XGBRFRegressor

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

################################################
# 1. Load Data
################################################
def load_data(dataframe_url):
    dataframe = pd.read_csv(dataframe_url)
    return dataframe



################################################
# 2. Exploratory Data Analysis
################################################

def check_df(dataframe):
    print('#######################  HEAD    #############################')
    print(dataframe.head(10))
    print('#######################  TAIL    ##############################')
    print(dataframe.tail(10))
    print('#######################  INFO    ##############################')
    print(dataframe.info())
    print('#######################  NULL SUM    ##############################')
    print(dataframe.isnull().sum())

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    temp_df[na_columns + '_NA_FLAG'] = np.where(temp_df[na_columns].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def msno(dataframe):
    msno.bar(dataframe)
    plt.show()

    msno.matrix(dataframe)
    plt.show()

    msno.heatmap(dataframe)
    plt.show()

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show(block=True)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}).sort_values(by='TARGET_MEAN',
                                                                                               ascending=False),
              end="\n\n\n")


################################################
# 3. Data Preprocessing And Feature Engineering
################################################
def analysis_func(dataframe):
    dataframe.drop(['Unnamed: 0', 'Model Name'], axis=1, inplace=True)
    # kolon isimlerini değiştirelim
    dataframe.columns = ['marka', 'islemci', 'isletim_sistemi', 'depolama', 'ram', 'ekran_boyutu',
                         'dokunmatik_ekran', 'fiyat']
    dataframe['depolama'] = [
        int(unit.split(' ')[0]) * 1024 if str(unit).endswith('TB') else str(unit).split(' ')[0] if str(unit).endswith(
            'GB') else unit for unit in dataframe['depolama']]
    # dataframe['depolama'] = dataframe['depolama'].astype(int)
    dataframe['ram'] = [str(unit).split(' ')[0] for unit in dataframe['ram']]
    dataframe['ram'] = dataframe['ram'].astype(int)
    dataframe['ekran_boyutu'] = [float(str(unit).split('cm (')[0]) / 2.54 for unit in dataframe['ekran_boyutu']]
    dataframe['dokunmatik_ekran'] = [1 if unit == 'Yes' else 0 for unit in dataframe['dokunmatik_ekran']]
    dataframe['depolama'] = [int(unit) if type(unit) in [int, str] else unit for unit in dataframe['depolama']]
    dataframe['ekran_boyutu'] = dataframe['ekran_boyutu'].astype(float)
    dataframe['ekran_boyutu'] = dataframe['ekran_boyutu'].apply(lambda x: '%.1f' % x)
    def price_func(dataframe):
        dataframe['fiyat'] = [round(int(unit.split('₹')[1].replace(',', '')) * 0.38) for unit in dataframe['fiyat']]
    price_func(dataframe)
    return dataframe



def grab_col_names(dataframe, cat_th=1, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
#cat_cols, num_cols, cat_but_car = grab_col_names(dff)
# cat_cols =  ['marka', 'islemci', 'isletim_sistemi']
# num_cols = ['depolama', 'ram', 'ekran_boyutu', 'dokunmatik_ekran', 'fiyat']

def missing_value_filling_function(dataframe):
    b = []
    for i in dataframe['ram']:
        a = dataframe[dataframe['ram'] == i]['depolama'].mode().values
        b.append(a[0])

    index_depolama_mod = list(zip(dff.index, b))
    for nan_inx, nan_dep in index_depolama_mod:
        dataframe.loc[nan_inx, 'depolama'] = nan_dep
#missing_value_filling_function(dff)


def creat_new_columns(dataframe):
    liste = []
    for col in dataframe['marka']:
        a = dataframe[dataframe['marka'] == col]['fiyat'].mean()
        liste.append(a)

    mean_series = pd.Series(liste).values
    dataframe['NEW_marka_mean'] = mean_series
    dataframe['NEW_marka_mean_rank'] = dataframe['NEW_marka_mean'].rank()

    dataframe['NEW_islemci'] = ['Intel' if unit.lower().split()[0] == 'core' or unit.lower().split()[0] == 'pentium' else \
                              'AMD' if unit.lower().split()[0] == 'ryzen' else \
                                  'MACOS' if len(unit.lower().split()[0]) >= 1 \
                                      else 'MediaTek' for unit in dataframe['islemci']]
    return dataframe

def outlier_thresholds(dataframe, col_name, q1=0.2, q3=0.98):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

'''for col in num_cols:
    outlier_thresholds(dff, col)'''

def check_outlier(dataframe, col_name, q1=0.2, q3=0.98):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def outlier_list(dataframe, num_cols):
    null_list = []
    for num_col in num_cols:
        outlier_true = check_outlier(dataframe, num_col)
        null_list.append(outlier_true)

    new_outlier_list = zip(num_cols, null_list)
    print(list(new_outlier_list))

#numerical_cols = ['depolama', 'ram', 'ekran_boyutu', 'dokunmatik_ekran', 'fiyat', 'NEW_marka_mean', 'NEW_marka_mean_rank']
#outlier_list(dff, numerical_cols)

#[('depolama', True), ('ram', True), ('ekran_boyutu', True), ('dokunmatik_ekran', False), ('fiyat', True), ('NEW_marka_mean', False), ('NEW_marka_mean_rank', False)]


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#replace_with_thresholds(dff, 'fiyat')

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#dff = one_hot_encoder(dff, cat_cols)
#cat_cols, num_cols, cat_but_car = grab_col_names(dff) # son değerleri tutuyoruz


################################################
# 4. Scale And Train Test Split
################################################

def scaler(dataframe, num_cols):
    scale = MinMaxScaler()
    scale_feats = scale.fit_transform(dataframe[num_cols])
    dff = pd.DataFrame(scale_feats, columns=num_cols)

    X = dff.drop(['fiyat'], axis=1)
    y = dff['fiyat']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    return X, y, X_train, X_test, y_train, y_test, dff

#X, y, X_train, X_test, y_train, y_test, dff = scaler(dff, num_cols)


def based_model(X_train, X_test, y_train, y_test, scoring='neg_mean_squared_error', scoring2='r2'):
    regressors = [('LR_model', LinearRegression()),
                  ('KNN_model', KNeighborsRegressor()),
                  ("CART_model", DecisionTreeRegressor()),
                  ("RF_model", RandomForestRegressor()),
                  ('Adaboost_model', AdaBoostRegressor()),
                  ('GBM_model', GradientBoostingRegressor()),
                  ('XGBoost_model', XGBRFRegressor())]

    for regressors_model_name, regressor_model in regressors:
        regressors_model_name = regressor_model
        regressors_model_name.fit(X_train, y_train)
        cv_mse = np.mean(-cross_val_score(regressor_model, X_test, y_test, cv=5, scoring='neg_mean_squared_error'))
        cv_r2 = np.mean(-cross_val_score(regressor_model, X_test, y_test, cv=5, scoring='r2'))
        print(f"{scoring}: {round(cv_mse, 13)} ({regressors_model_name})\n{scoring2}: {round(cv_r2, 13)} ({regressors_model_name})")
#based_model(X_train, X_test, y_train, y_test)


########################################################
# 5. Automated Hyperparameter Optimization
########################################################
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

gbm_params = {'n_estimators': [100, 200, 500, 1000],
              'learning_rate': [0.1, 0.01, 0.05],
              'max_depth': [3, 4, 5],
              'min_samples_split': [2, 5, 10]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

regressors = [('KNN_model', KNeighborsRegressor(), knn_params),
               ("CART_model", DecisionTreeRegressor(), cart_params),
               ("RF_model", RandomForestRegressor(), rf_params),
               ('GBM_model', GradientBoostingRegressor(), gbm_params),
               ('XGBoost_model', XGBRFRegressor(), xgboost_params)]

def hyperparameter_optimization(X_train, X_test, y_train, y_test, cv=5, scoring="neg_mean_squared_error", scoring2='r2'):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        name = regressor.fit(X_train, y_train)
        cv_mse = np.mean(cross_val_score(name, X_test, y_test, cv=3, scoring=scoring))
        print(f"{scoring} (Before): {round(cv_mse, 10)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_mse = np.mean(-cross_val_score(final_model, X_test, y_test, cv=3, scoring=scoring))

        print(f"{scoring} (After): {round(cv_mse, 10)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

#best_models = hyperparameter_optimization(X_train, X_test, y_train, y_test)

########################################################
# 6. Stacking & Ensemble Learning
########################################################
def voting_regressor(best_models, X_train, X_test, y_train, y_test, scoring='neg_mean_squared_error'):
    print("Voting Classifier...")
    keys = list(best_models.keys())
    voting_clf = VotingRegressor(estimators=[('KNN', keys[0]),
                                            ('RF', keys[2]),
                                            ('GBM', keys[3])]).fit(X_train, y_train)
    cv_results = cross_val_score(voting_clf, X_test, y_test, cv=5, scoring=scoring)
    print(f"MSE: {-cv_results.mean()}")
    #print(f"R2: {cv_results['r2'].mean()}")
    return voting_clf

#voting_clf = voting_regressor(best_models, X_train, X_test, y_train, y_test)


################################################
# 7.Pipeline Main Function
################################################
def main():
    dff = load_data(dataframe_url='datasets/8-laptopPrediction/Laptops.csv')
    dff = analysis_func(dff)
    cat_cols, num_cols, cat_but_car = grab_col_names(dff)
    missing_value_filling_function(dff)
    dff = creat_new_columns(dff)
    cat_cols, num_cols, cat_but_car = grab_col_names(dff)
    for col in num_cols:
        outlier_thresholds(dff, col)
    check_outlier(dff, num_cols)
    replace_with_thresholds(dff, 'fiyat')
    dff = one_hot_encoder(dff, cat_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(dff)
    X, y, X_train, X_test, y_train, y_test, dff = scaler(dff, num_cols)
    based_model(X_train, X_test, y_train, y_test)
    best_models = hyperparameter_optimization(X_train, X_test, y_train, y_test)
    voting_clf = voting_regressor(best_models, X_train, X_test, y_train, y_test)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf

if __name__ == "__main__":
    print("İşlem başladı")
    main()

