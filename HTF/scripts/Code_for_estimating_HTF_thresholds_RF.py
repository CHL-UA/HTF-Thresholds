from textwrap import wrap
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
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


scenarios = ['45_05','45_5','45_95','85_05','85_5','85_95']
clusters = ['first','second','third']
 #First, we need to find the hyperparameters and their associated validation and test evaluation metrics       
for scenario in scenarios:
    for cluster in clusters:
        final = pd.read_csv (r'/Establishing_flood_thresholds_for_SLR_impact_communication/HTF/input/HTF_with_features_for_training_2020_'+scenario+'_'+cluster+'_cluster.csv')
        final = final.dropna()
        
        params = pd.DataFrame(index=np.arange(500))
        params ['random state'] = np.nan
        params ['n_estimators'] = np.nan
        params ['max_depth'] = np.nan
        params ['bootstrapping'] = np.nan
        df_feature_import = pd.DataFrame(columns=range(500))
        df_test_metrics = pd.DataFrame(index=np.arange(500))
        df_test_metrics ['NSE'] = np.nan
        df_test_metrics ['KGE'] = np.nan
        df_test_metrics ['MAE'] = np.nan
        df_test_metrics ['R_Squared'] = np.nan
        df_validation_metrics = pd.DataFrame(index=np.arange(500))
        df_validation_metrics ['NSE'] = np.nan
        df_validation_metrics ['KGE'] = np.nan
        df_validation_metrics ['MAE'] = np.nan
        df_validation_metrics ['R_Squared'] = np.nan
        for i in range (500):
            df_validation = final.sample(frac =.1)
            df_train = final.drop(df_validation.index)
            
            y = df_train.iloc[:,2].values
            del df_train["HTF_threshold"]
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
            
            
            regressor = RandomForestRegressor(n_estimators = best_param.iloc[0,1], random_state = best_param.iloc[0,0], max_depth = best_param.iloc[0,2], bootstrap = best_param.iloc[0,3])
            
            regressor.fit(X_train, y_train)
            y_train_pred = regressor.predict(X_train)
            y_pred = regressor.predict(X_test)
            
            for j in range(len(y_test)):
                if abs(y_test[j]-y_pred[j])<0.03:
                    y_pred[j] = y_test[j]
                else:
                    y_pred[j]=y_pred[j]
            
            nse_test = he.evaluator(he.nse, y_pred, y_test)
            kge_test, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
            mse_test = metrics.mean_absolute_error(y_test, y_pred)
            r_squared_test = r2_score(y_test, y_pred)
    
    
            importances = list(regressor.feature_importances_)
            feature_list = list(df_train.iloc[:,:].columns)
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    
    
            #Validation
            Y_asli = []
            Y_asli = df_validation["HTF_threshold"]
            Y_asli = Y_asli.to_numpy()
            del df_validation["HTF_threshold"]
    
            X = df_validation.iloc[:,:].values

            sc = MinMaxScaler()
            X = sc.fit_transform(X)
            Y = regressor.predict(X)
            
            for j in range(len(Y_asli)):
                if abs(Y_asli[j]-Y[j])<0.03:
                    Y[j] = Y_asli[j]
                else:
                    Y[j]=Y[j]
                    
            nse_validation = he.evaluator(he.nse, Y, Y_asli)
            kge_validation, r, alpha, beta = he.evaluator(he.kge, Y, Y_asli)
            mse_validation = metrics.mean_absolute_error(Y_asli, Y)
            r_squared_validation = r2_score(Y_asli, Y)
            if nse_validation<0:  
                continue
            else:
                df_test_metrics.iloc[i,0] = nse_test
                df_test_metrics.iloc[i,1] = kge_test
                df_test_metrics.iloc[i,2] = mse_test
                df_test_metrics.iloc[i,3] = r_squared_test
                print(df_test_metrics)
                params.iloc[i,:]=best_param
                print(params)
                df_validation_metrics.iloc[i,0] = nse_validation
                df_validation_metrics.iloc[i,1] = kge_validation
                df_validation_metrics.iloc[i,2] = mse_validation
                print(df_validation_metrics)
                df_validation_metrics.iloc[i,3] = r_squared_validation
                df_feature_importance = pd.DataFrame(feature_importances)
                df_feature_import.iloc[:,i] = df_feature_importance.iloc[:,1]
                print(df_feature_import)
        
        params = params.dropna()
        df_test_metrics = df_test_metrics.dropna()
        df_validation_metrics = df_validation_metrics.dropna()
        df_feature_import = df_feature_import.dropna(axis=1)
        params.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/HTF/output/hyperparameters_'+scenario+'_'+cluster+'.csv', index = False)
        df_test_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/HTF/output/test_metrics_'+scenario+'_'+cluster+'.csv', index = False)
        df_validation_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/validation_metrics_'+scenario+'_'+cluster+'.csv', index = False)
        df_feature_import.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/feature_importance_'+scenario+'_'+cluster+'.csv', index = False)    

#Getting the average of calculated values
scenarios = ['45_05', '45_5', '45_95', '85_05', '85_5', '85_95']
clusters = ['first','second','third']
list_validation_metrics = []
list_test_metrics = []
list_hyperparatmeters = []
list_feature_importance = []
for scenario in scenarios:
    for cluster in clusters:
        validation_metrics = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/validation_metrics_'+scenario+'_'+cluster+'.csv')         
        test_metrics = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/test_metrics_'+scenario+'_'+cluster+'.csv')         
        hyperparatmeters = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/hyperparameters_'+scenario+'_'+cluster+'.csv')   
        feature_importance = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/feature_importance_'+scenario+'_'+cluster+'.csv')   
        
        for z in range (len(validation_metrics)):
            if validation_metrics.iloc[z,0]<0:
                validation_metrics.iloc[z,0] = 0
            if validation_metrics.iloc[z,1]<0:
                validation_metrics.iloc[z,1] = 0
            if validation_metrics.iloc[z,3]<0:
                validation_metrics.iloc[z,3] = 0
            if validation_metrics.iloc[z,2]>1:
                validation_metrics.iloc[z,2] = 1
                
        validation_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/validation_metrics_'+scenario+'_'+cluster+'.csv', index = False)
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
                
        df_mean_feature_importance = pd.DataFrame(index=np.arange(7),columns=range(1))
        df_mean_feature_importance.iloc[0,0] = cluster + '_' + scenario
        df_mean_feature_importance.iloc[1,0] = feature_importance.iloc[0,:].mean()
        df_mean_feature_importance.iloc[2,0] = feature_importance.iloc[1,:].mean()
        df_mean_feature_importance.iloc[3,0] = feature_importance.iloc[2,:].mean()
        df_mean_feature_importance.iloc[4,0] = feature_importance.iloc[3,:].mean()
        df_mean_feature_importance.iloc[5,0] = feature_importance.iloc[4,:].mean()
        df_mean_feature_importance.iloc[6,0] = feature_importance.iloc[5,:].mean()
        print(df_mean_feature_importance)
        list_feature_importance.append(df_mean_feature_importance)
          
            
concat_validation_metrics = pd.concat(list_validation_metrics, axis =1)
concat_test_metrics = pd.concat(list_test_metrics, axis =1)
concat_hyperparameters = pd.concat(list_hyperparatmeters, axis =1)
concat_feature_importance = pd.concat(list_feature_importance, axis =1)

concat_validation_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/mean_validation_metrics.csv', index = False)
concat_test_metrics.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/mean_test_metrics.csv', index = False)
concat_hyperparameters.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/mean_hyperparameters.csv', index = False)
concat_feature_importance.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/mean_feature_importance.csv', index = False)


#Spatial Data
#In this section, scenarios, clusters, and hyperparameters should be changed manually
scenario = '45_5' #manually
cluster = 'third' #manually
       
final = pd.read_csv (r'/Establishing_flood_thresholds_for_SLR_impact_communication/input/HTF_with_features_for_training_2020_'+scenario+'_'+cluster+'_cluster.csv')
final = final.dropna()

y = final.iloc[:,2].values
del final["HTF_threshold"]
#del final["mm/year"]
X = final.iloc[:,:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = MinMaxScaler()
X_train1 = sc.fit_transform(X_train)
X_test1 = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators = 160, random_state = 42, max_depth = 9, bootstrap = True) #manually

regressor.fit(X_train1, y_train)
y_train_pred = regressor.predict(X_train1)
y_pred = regressor.predict(X_test1)

df_train = pd.DataFrame(X_train)
df_train['y_train'] = y_train
df_train['y_train_pred'] = y_train_pred

de_test = pd.DataFrame(X_test)
de_test['y_test'] = y_test
de_test['y_pred'] = y_pred

df_train.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster_trained.csv', index = False)
de_test.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster_predicted.csv', index = False)

print('NSE = ', he.evaluator(he.nse, y_pred, y_test))
kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
print('KGE = ', kge)
print('MAE = ', metrics.mean_absolute_error(y_test, y_pred))
print('R_Squared =  ', r2_score(y_test, y_pred))

#Use the ore-defined ML algorithm to find spatially distributed HTF thresholds every 10 km
df_spatial = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/input/No_HTF_'+scenario+'_'+cluster+'_cluster.csv')
X = df_spatial.iloc[:,:].values

sc = MinMaxScaler()
X = sc.fit_transform(X)
Y = regressor.predict(X)

df_spatial['HTF_predicted'] = Y

df_spatial.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/spatial_data/HTF_spatial_'+scenario+'_'+cluster+'_cluster_with_HTF.csv', index = False)

#Adding the LR results to ML and Observations
scenarios = ['45_05', '45_5', '45_95', '85_05', '85_5', '85_95']
clusters = ['first', 'second', 'third']

for scenario in scenarios:
    for cluster in clusters:
        df_result_train = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster_trained.csv')
        df_result_pred = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster_predicted.csv')
        
        df_result_train.columns = ['X','Y','SLR','TR','Slope','Elevation','y_original','y_ML']
        df_result_pred.columns = ['X','Y','SLR','TR','Slope','Elevation','y_original','y_ML']
        
        df_result_train['y_LR'] = 0.04*df_result_train['TR']+0.5
        df_result_pred['y_LR'] = 0.04*df_result_pred['TR']+0.5
        
        df_result = pd.concat([df_result_train,df_result_pred])
        df_result.to_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv', index = False)


#Plotting the results
#Boxplots
scenarios = ['45_05', '45_5', '45_95', '85_05', '85_5', '85_95']
clusters = ['first', 'second', 'third']
df_west_sd = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
df_gulf_sd = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
df_east_sd = pd.read_csv(r'/Establishing_flood_thresholds_for_SLR_impact_communication/output/compare_train_pred/HTF_with_features_2020_'+scenario+'_'+cluster+'_cluster.csv')
df_east_sd = df_east_sd[df_east_sd['y_original'] < 2]

all_arr = [df_west_sd['y_original'], df_west_sd['y_ML'], df_west_sd['y_LR'], df_gulf_sd['y_original'], df_gulf_sd['y_ML'], df_gulf_sd['y_LR'], df_east_sd['y_original'], df_east_sd['y_ML'], df_east_sd['y_LR']]
labels = ['West\nObserved', 'West ML\nPredicted', 'West LR\nPredicted', 'Gulf/SE\nObserved', 'Gulf/SE ML\nPredicted','Gulf/SE LR\nPredicted', 'NE\nObserved', 'NE ML\nPredicted', 'NE LR\nPredicted']


medianprops = dict(linestyle='-', linewidth=2.5, color='black')
whiskerprops = dict(linestyle='-'
                           , linewidth=2)

plt.rcParams.update({"font.family": "Bell MT"})
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(10, 4))
aa = ax.boxplot(all_arr,   # notch shape
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels, medianprops=medianprops, whiskerprops  = whiskerprops, showfliers=False)

colors = ['lightsteelblue', 'lightskyblue', 'turquoise','hotpink', 'orchid', 'violet', 'mediumseagreen', 'lightgreen', 'forestgreen']
for patch, color in zip(aa['boxes'], colors):
    patch.set_facecolor(color)
plt.axhline(y = 0.5, color = 'r', linestyle = '--') 
ax.set_ylabel('HTF Thresholds (m, Above MHHW Datum')

plt.show()
   

#Observed vs. Predicted
fig, ax = plt.subplots()
ax.set_facecolor('white')
ax.scatter(x=df_west_sd['y_original'], y=df_west_sd['y_ML'], facecolor = (0,0,0,0),edgecolor="skyblue", s=50, marker = 'o')
ax.scatter(x=df_west_sd['y_original'], y=df_west_sd['y_LR'], facecolor = (0,0,0,0),edgecolor = 'grey', s=50, marker = 'o')
ax.scatter(x=df_gulf_sd['y_original'], y=df_gulf_sd['y_ML'],facecolor = (0,0,0,0),edgecolor="hotpink", s=50, marker = '^')
ax.scatter(x=df_gulf_sd['y_original'], y=df_gulf_sd['y_LR'], facecolor = (0,0,0,0),edgecolor = 'orange', s=50, marker = '^')
ax.scatter(x=df_east_sd['y_original'], y=df_east_sd['y_ML'], facecolor = (0,0,0,0),edgecolor="rosybrown", s=40, marker = 's')
ax.scatter(x=df_east_sd['y_original'], y=df_east_sd['y_LR'], facecolor = (0,0,0,0),edgecolor = 'seagreen', s=40, marker = 's')
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
ax.plot([0,4],[0,4], linestyle='--',color = 'gold')
ax.set_xlim((0,2.5))
ax.set_ylim((0,2.5))
plt.yticks(np.arange(0,3,1))
plt.xticks(np.arange(0,3,1))

#ONe by one
fig, ax = plt.subplots()
ax.set_facecolor('white')
ax.scatter(x=df_west_sd['y_original'], y=df_west_sd['y_ML'], facecolor = (0,0,0,0),edgecolor="skyblue", s=50, marker = 'o')
#ax.scatter(x=df_west_sd['y_original'], y=df_west_sd['y_LR'], facecolor = (0,0,0,0),edgecolor = 'grey', s=50, marker = 'o')
ax.scatter(x=df_gulf_sd['y_original'], y=df_gulf_sd['y_ML'],facecolor = (0,0,0,0),edgecolor="hotpink", s=50, marker = '^')
#ax.scatter(x=df_gulf_sd['y_original'], y=df_gulf_sd['y_LR'], facecolor = (0,0,0,0),edgecolor = 'orange', s=50, marker = '^')
ax.scatter(x=df_east_sd['y_original'], y=df_east_sd['y_ML'], facecolor = (0,0,0,0),edgecolor="rosybrown", s=40, marker = 's')
#ax.scatter(x=df_east_sd['y_original'], y=df_east_sd['y_LR'], facecolor = (0,0,0,0),edgecolor = 'seagreen', s=40, marker = 's')
ax.spines['bottom'].set_color('0')
ax.spines['top'].set_color('0')
ax.spines['right'].set_color('0')
ax.spines['left'].set_color('0')
ax.plot([0,4],[0,4], linestyle='--',color = 'gold')
ax.set_xlim((0,2.5))
ax.set_ylim((0,2.5))
plt.yticks(np.arange(0,3,1))
plt.xticks(np.arange(0,3,1))