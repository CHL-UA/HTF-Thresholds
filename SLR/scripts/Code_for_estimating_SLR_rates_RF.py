import seaborn as sns
import hydroeval as he
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


scenarios = ['45_05', '45_5', '45_95', '85_05', '85_5', '85_95']
clusters = ['first', 'second', 'third']

#First, we need to find the hyperparameters and their associated validation and test evaluation metrics
for scenario in scenarios:
    for cluster in clusters:
        final = pd.read_csv (r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/input/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
        final = final.dropna()
        
        params = pd.DataFrame(index=np.arange(100))
        params ['random state'] = np.nan
        params ['n_estimators'] = np.nan
        params ['max_depth'] = np.nan
        params ['bootstrapping'] = np.nan
        df_feature_import = pd.DataFrame(columns=range(100))
        df_test_metrics = pd.DataFrame(index=np.arange(100))
        df_test_metrics ['NSE'] = np.nan
        df_test_metrics ['KGE'] = np.nan
        df_test_metrics ['MAE'] = np.nan
        df_test_metrics ['R_Squared'] = np.nan
        df_validation_metrics = pd.DataFrame(index=np.arange(100))
        df_validation_metrics ['NSE'] = np.nan
        df_validation_metrics ['KGE'] = np.nan
        df_validation_metrics ['MAE'] = np.nan
        df_validation_metrics ['R_Squared'] = np.nan
        for i in range (100):
            df_validation = final.sample(frac =.2)
            df_train = final.drop(df_validation.index)
            
            y = df_train.iloc[:,2].values
            del df_train["SLR"]
            X = df_train.iloc[:,:].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            sc = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            #Randomized Search
            regressor = RandomForestRegressor()
            random_grid = {'n_estimators':[50,60,75,100,200,300,400],
                            'max_depth':[2,3,4,5,7,8,9,10],
                            'bootstrap':[True, False],
                            'random_state':[10, 42, 50,60]}
            clf = RandomizedSearchCV (regressor, random_grid, n_iter = 100, verbose = 2, random_state = 42, n_jobs = -1)
            search = clf.fit(X_train, y_train)
            best_param = pd.DataFrame.from_dict([search.best_params_])
            params.iloc[i,:]=best_param
            
            regressor = RandomForestRegressor(n_estimators = best_param.iloc[0,1], random_state = best_param.iloc[0,0], max_depth = best_param.iloc[0,2], bootstrap = best_param.iloc[0,3])
            
            regressor.fit(X_train, y_train)
            y_train_pred = regressor.predict(X_train)
            y_pred = regressor.predict(X_test)
            
            nse_test = he.evaluator(he.nse, y_pred, y_test)
            kge_test, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
            mse_test = metrics.mean_absolute_error(y_test, y_pred)
            r_squared_test = r2_score(y_test, y_pred)
            df_test_metrics.iloc[i,0] = nse_test
            df_test_metrics.iloc[i,1] = kge_test
            df_test_metrics.iloc[i,2] = mse_test
            df_test_metrics.iloc[i,3] = r_squared_test
            print(df_test_metrics)
            
            importances = list(regressor.feature_importances_)
            feature_list = list(df_train.iloc[:,:].columns)
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
            df_feature_importance = pd.DataFrame(feature_importances)
            df_feature_import.iloc[:,i] = df_feature_importance.iloc[:,1]
            print(df_feature_import)
            
            #Validation
            Y_asli = []
            Y_asli = df_validation["SLR"]
            del df_validation["SLR"]
            
            X = df_validation.iloc[:,:].values

            sc = MinMaxScaler()
            X = sc.fit_transform(X)
            Y = regressor.predict(X)
            
            nse_validation = he.evaluator(he.nse, Y_asli, Y)
            kge_validation, r, alpha, beta = he.evaluator(he.kge, Y_asli, Y)
            mse_validation = metrics.mean_absolute_error(Y_asli, Y)
            r_squared_validation = r2_score(Y_asli, Y)
            df_validation_metrics.iloc[i,0] = nse_validation
            df_validation_metrics.iloc[i,1] = kge_validation
            df_validation_metrics.iloc[i,2] = mse_validation
            df_validation_metrics.iloc[i,3] = r_squared_validation
            print(df_validation_metrics)
            
        params.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/hyperparameters_'+scenario+'_'+cluster+'.csv', index = False)
        df_test_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/test_metrics_'+scenario+'_'+cluster+'.csv', index = False)
        df_validation_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/validation_metrics_'+scenario+'_'+cluster+'.csv', index = False)
        df_feature_import.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/feature_importance_'+scenario+'_'+cluster+'.csv', index = False)
        
            
            
            
#Getting the average of calculated values
scenarios = ['45_05', '45_5', '45_95', '85_05', '85_5', '85_95']
clusters = ['first', 'second', 'third']
list_validation_metrics = []
list_test_metrics = []
list_hyperparatmeters = []
list_feature_importance = []
for scenario in scenarios:
    for cluster in clusters:
        validation_metrics = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/validation_metrics_'+scenario+'_'+cluster+'.csv')         
        test_metrics = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/test_metrics_'+scenario+'_'+cluster+'.csv')         
        hyperparatmeters = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/hyperparameters_'+scenario+'_'+cluster+'.csv')   
        feature_importance = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/feature_importance_'+scenario+'_'+cluster+'.csv')   
        
        df_mean_validation_metrics = pd.DataFrame(index=np.arange(1))
        df_mean_validation_metrics[cluster + '_' + scenario+'_nse'] = np.nan
        df_mean_validation_metrics[cluster + '_' + scenario+'_kge'] = np.nan
        df_mean_validation_metrics[cluster + '_' + scenario+'_mae'] = np.nan
        df_mean_validation_metrics[cluster + '_' + scenario+'_r_squared'] = np.nan
        df_mean_validation_metrics.iloc[0,0] = validation_metrics['NSE'].mean()
        df_mean_validation_metrics.iloc[0,1] = validation_metrics['KGE'].mean()
        df_mean_validation_metrics.iloc[0,2] = validation_metrics['MAE'].mean()
        df_mean_validation_metrics.iloc[0,3] = validation_metrics['R_Squared'].mean()
        print(df_mean_validation_metrics)
        list_validation_metrics.append(df_mean_validation_metrics)
        
        df_mean_test_metrics = pd.DataFrame(index=np.arange(1))
        df_mean_test_metrics[cluster + '_' + scenario+'_nse'] = np.nan
        df_mean_test_metrics[cluster + '_' + scenario+'_kge'] = np.nan
        df_mean_test_metrics[cluster + '_' + scenario+'_mae'] = np.nan
        df_mean_test_metrics[cluster + '_' + scenario+'_r_squared'] = np.nan
        df_mean_test_metrics.iloc[0,0] = test_metrics['NSE'].mean()
        df_mean_test_metrics.iloc[0,1] = test_metrics['KGE'].mean()
        df_mean_test_metrics.iloc[0,2] = test_metrics['MAE'].mean()
        df_mean_test_metrics.iloc[0,3] = test_metrics['R_Squared'].mean()
        print(df_mean_test_metrics)
        list_test_metrics.append(df_mean_test_metrics)
        
        df_mean_hyperparatmeters = pd.DataFrame(index=np.arange(1))
        df_mean_hyperparatmeters[cluster + '_' + scenario+'_random state'] = np.nan
        df_mean_hyperparatmeters[cluster + '_' + scenario+'_n_estimators'] = np.nan
        df_mean_hyperparatmeters[cluster + '_' + scenario+'_max_depth'] = np.nan
        df_mean_hyperparatmeters.iloc[0,0] = hyperparatmeters['random state'].mean()
        df_mean_hyperparatmeters.iloc[0,1] = hyperparatmeters['n_estimators'].mean()
        df_mean_hyperparatmeters.iloc[0,2] = hyperparatmeters['max_depth'].mean()
        print(df_mean_hyperparatmeters)
        list_hyperparatmeters.append(df_mean_hyperparatmeters)
                
        df_mean_feature_importance = pd.DataFrame(index=np.arange(10),columns=range(1))
        df_mean_feature_importance.iloc[0,0] = cluster + '_' + scenario
        df_mean_feature_importance.iloc[1,0] = feature_importance.iloc[0,:].mean()
        df_mean_feature_importance.iloc[2,0] = feature_importance.iloc[1,:].mean()
        df_mean_feature_importance.iloc[3,0] = feature_importance.iloc[2,:].mean()
        df_mean_feature_importance.iloc[4,0] = feature_importance.iloc[3,:].mean()
        df_mean_feature_importance.iloc[5,0] = feature_importance.iloc[4,:].mean()
        df_mean_feature_importance.iloc[6,0] = feature_importance.iloc[5,:].mean()
        df_mean_feature_importance.iloc[7,0] = feature_importance.iloc[6,:].mean()
        df_mean_feature_importance.iloc[8,0] = feature_importance.iloc[7,:].mean()
        df_mean_feature_importance.iloc[9,0] = feature_importance.iloc[8,:].mean()
        print(df_mean_feature_importance)
        list_feature_importance.append(df_mean_feature_importance)
          
            
concat_validation_metrics = pd.concat(list_validation_metrics, axis =1)
concat_test_metrics = pd.concat(list_test_metrics, axis =1)
concat_hyperparameters = pd.concat(list_hyperparatmeters, axis =1)
concat_feature_importance = pd.concat(list_feature_importance, axis =1)

concat_validation_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output//mean_validation_metrics.csv', index = False)
concat_test_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/mean_test_metrics.csv', index = False)
concat_hyperparameters.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/mean_hyperparameters.csv', index = False)
concat_feature_importance.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/mean_feature_importance.csv', index = False)



#Spatial Data
#In this section, scenarios, clusters, and hyperparameters should be changed manually
scenario = '45_5' #manually
cluster = 'third' #manually
       
final = pd.read_csv (r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/input/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
final = final.dropna()

y = final.iloc[:,2].values
del final["SLR"]
X = final.iloc[:,:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = MinMaxScaler()
X_train1 = sc.fit_transform(X_train)
X_test1 = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators = 140, random_state = 42, max_depth = 10, bootstrap = True) #manually

regressor.fit(X_train1, y_train)
#Get the values predicted for places we have SLR rates available
y_train_pred = regressor.predict(X_train1)
y_pred = regressor.predict(X_test1)

df_train = pd.DataFrame(X_train)
df_train['y_train'] = y_train
df_train['y_train_pred'] = y_train_pred

de_test = pd.DataFrame(X_test)
de_test['y_test'] = y_test
de_test['y_pred'] = y_pred

df_train.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster_trained.csv', index = False)
de_test.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster_predicted.csv', index = False)

print('NSE = ', he.evaluator(he.nse, y_pred, y_test))
kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
print('KGE = ', kge)
print('MAE = ', metrics.mean_absolute_error(y_test, y_pred))
print('R_Squared =  ', r2_score(y_test, y_pred))

#Use the pre-defined ML algorithm to define SLR rates for places we have HTF thresholds available for the next step which is finding HTF thresholds
df_spatial_HTF = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/input/No_SLR_with_features_'+cluster+'_cluster.csv')
X = df_spatial_HTF.iloc[:,:].values

sc = MinMaxScaler()
X = sc.fit_transform(X)
Y = regressor.predict(X)

df_spatial_HTF['SLR_predicted'] = Y
df_spatial_HTF['mm/year'] = df_spatial_HTF['SLR_predicted']*10/21
df_spatial_HTF.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/for_HTF/SLR_spatial_'+scenario+'_'+cluster+'_cluster_with_SLR_for_HTF.csv', index = False)

#Use the pre-defined ML algorithm to find spatially distributed SLR rates every 10 km
df_spatial = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/input'+'//'+cluster+'_cluster_no_SLR.csv')
X = df_spatial.iloc[:,:].values

sc = MinMaxScaler()
X = sc.fit_transform(X)
Y = regressor.predict(X)

df_spatial['SLR_predicted'] = Y
df_spatial['mm/year'] = df_spatial['SLR_predicted']*10/21
df_spatial.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/spatial_data/SLR_spatial_'+scenario+'_'+cluster+'_cluster_with_SLR.csv', index = False)


#Concatenate pred and train csvs from training process
scenarios = ['45_05', '45_5', '45_95', '85_05', '85_5', '85_95']
clusters = ['first', 'second', 'third']

for scenario in scenarios:
    for cluster in clusters:
        df_result_train = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster_trained.csv')
        df_result_pred = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster_predicted.csv')
        
        df_result_train.columns = ['X','Y','OC','GH','Salt','MSLP','SP','SST','VLM','y_original','y_ML']
        df_result_pred.columns = ['X','Y','OC','GH','Salt','MSLP','SP','SST','VLM','y_original','y_ML']
        df_result_train['y_original'] = df_result_train['y_original']*10/21
        df_result_train['y_ML'] = df_result_train['y_ML']*10/21
        df_result_pred['y_original'] = df_result_pred['y_original']*10/21
        df_result_pred['y_ML'] = df_result_pred['y_ML']*10/21
        
        df_result = pd.concat([df_result_train,df_result_pred])
        df_result.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv', index = False)


#Plotting the results
#Boxplots
scenarios = ['45_05', '45_5', '45_95', '85_05', '85_5', '85_95']
clusters = ['first', 'second', 'third']
df_west_sd = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
df_gulf_sd = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
df_east_sd = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/SLR/output/compare_train_test/SLR_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
  
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 7
plt.rcParams['axes.unicode_minus'] = False
my_pal = {"y_original": "lightsteelblue", "y_ML":"skyblue"}

plt.figure(figsize=(3, 2.5))
ax = sns.violinplot(data=df_west_sd[['y_original','y_ML']], palette=my_pal, inner=None)

for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width/2, height,
                       transform=ax.transData))
    
num_items = len(ax.collections)
for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.15)
    
sns.boxplot(data=df_west_sd[['y_original','y_ML']], width=0.25,
            showfliers=False, showmeans=True, 
            meanprops=dict(marker='o', markerfacecolor='maroon',
                           markersize=10, zorder=3),
            boxprops=dict(facecolor=(0,0,0,0), 
                          linewidth=3, zorder=3),
            whiskerprops=dict(linewidth=3),
            capprops=dict(linewidth=3),
            medianprops=dict(linewidth=3))
ax.set_xticklabels(['West\nObserved', 'West ML\nPredicted'], size=7)
ax.set_ylabel("SLR Rates (mm/year)", size=7)
plt.yticks(np.arange(-2, 8, 2), size=7)
plt.show()

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 7
plt.rcParams['axes.unicode_minus'] = False
my_pal = {"y_original": "hotpink", "y_ML":"orchid"}

plt.figure(figsize=(3,2.5))
ax = sns.violinplot(data=df_gulf_sd[['y_original','y_ML']], palette=my_pal, inner=None)

for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width/2, height,
                       transform=ax.transData))
    
num_items = len(ax.collections)
for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.15)
    
sns.boxplot(data=df_gulf_sd[['y_original','y_ML']], width=0.25,
            showfliers=False, showmeans=True, 
            meanprops=dict(marker='o', markerfacecolor='maroon',
                           markersize=10, zorder=3),
            boxprops=dict(facecolor=(0,0,0,0), 
                          linewidth=3, zorder=3),
            whiskerprops=dict(linewidth=3),
            capprops=dict(linewidth=3),
            medianprops=dict(linewidth=3))
ax.set_xticklabels(['Gulf/SE\nObserved', 'Gulf/SE ML\nPredicted'], size=7)
ax.set_ylabel("SLR Rates (mm/year)", size=7)
plt.yticks(np.arange(2, 14, 2), size=7)
plt.show()

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 7
plt.rcParams['axes.unicode_minus'] = False
my_pal = {"y_original": "mediumseagreen", "y_ML":"lightgreen"}

plt.figure(figsize=(3, 2.5))
ax = sns.violinplot(data=df_east_sd[['y_original','y_ML']], palette=my_pal, inner=None)

for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width/2, height,
                       transform=ax.transData))
    
num_items = len(ax.collections)
for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.15)
    
sns.boxplot(data=df_east_sd[['y_original','y_ML']], width=0.25,
            showfliers=False, showmeans=True, 
            meanprops=dict(marker='o', markerfacecolor='maroon',
                           markersize=10, zorder=3),
            boxprops=dict(facecolor=(0,0,0,0), 
                          linewidth=3, zorder=3),
            whiskerprops=dict(linewidth=3),
            capprops=dict(linewidth=3),
            medianprops=dict(linewidth=3))
ax.set_xticklabels(['NE\nObserved', 'NE ML\nPredicted'], size=7)
ax.set_ylabel("SLR Rates (mm/year)", size=7)
plt.yticks(np.arange(3, 8, 1), size=7)
plt.show()

#Observed vs. ML predicted
fig, ax = plt.subplots()
ax.set_facecolor('white')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 7
#ax2=ax.twinx()
ax.scatter(x = df_west_sd['y_original'], y = df_west_sd['y_ML'], facecolor = (0,0,0,0),edgecolor="blue", s=50, marker = 'o')
ax.scatter(x = df_gulf_sd['y_original'], y = df_gulf_sd['y_ML'],facecolor = (0,0,0,0),edgecolor="hotpink", s=50, marker = '^')
ax.scatter(x = df_east_sd['y_original'], y = df_east_sd['y_ML'], facecolor = (0,0,0,0),edgecolor="limegreen", s=40, marker = 's')









































