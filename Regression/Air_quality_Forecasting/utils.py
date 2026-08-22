# Air quality Forecasting utility function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit,cross_val_score

#  Data Loading

def load_data(filepath = 'data/PRSA_data.csv'):
    df = pd.read_csv(filepath)
    return df

df = load_data('data/air_quality_features.csv')
df.head()
# Feature Engineering

def create_feature(df):
    df['Datetime'] = pd.to_datetime(df[['year','month','day','hour']])
    df = df.sort_values("Datetime").reset_index(drop = True)

    # Calender features
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['quarter'] = df['Datetime'].dt.quarter
    df['dayofyear'] = df['Datetime'].dt.dayofyear
    df['is_weekend'] = (df['dayofweek']>=5).astype(int)
    df[['hour','dayofweek','quarter','dayofyear','is_weekend']].head()

    # Lag features
    df['lag_1'] = df['pm2.5'].shift(1) 
    df['lag_24'] = df['pm2.5'].shift(24) 
    # Rolling features
    df['roll_24'] = df['pm2.5'].shift(1).rolling(Window=24).mean() 
    df['roll_168'] = df['pm2.5'].shift(1).rolling(Window=168).mean() 

    # One hot Encoding(categorical)
    df = pd.get_dummies(df,columns=['cbwd'],prefix='wind',drop_first=True)
    wind_cols = [c for c in df.columns if c.startswith('wind_')]
    df['wind_cols'] = df['wind_cols'].astype(int)

    # Drop rows missing target
    df = df.dropna(subset=['pm2.5','lag_1','lag_24','roll_24','roll_168'])
    df = df.reset_index(drop=True)
    return df


# Model Evaluation
def evaluate_model(name,y_test,y_pred):
    metrics = {
        'Model': name,
        'RMSE' : float(np.sqrt(mean_squared_error(y_test,y_pred))),
        'MAE' : float(mean_absolute_error(y_test,y_pred)),
        'R2' : float(r2_score(y_test,y_pred)),
        'MAPE' : float(mean_absolute_percentage_error(y_test,y_pred))}

    print(f"\n{'='*40}\n {name}\n{'='*40}")
    for k,v in metrics.items():
        if k != 'Model':
            print(f"{k:6s}:{v:.4f}")
    return metrics

def plot_prediction_model(y_test,y_pred,name,n=500):
    fig,ax = plt.subplots(figsize=(14,4))
    ax.plot(np.array(y_test)[:n], label='Actual', color = 'steelblue',linewidth=1.2)
    ax.plot(np.array(y_test)[:n], label='predicted', color = 'tomato',linewidth=1.0,alpha=0.85)    
    ax.set_title(f"Predicted vs Actual-{name} (fist {n} test rows)")
    ax.set_xlabel('Hour Index (test set)')    
    ax.set_ylabel('PM2.5')
    ax.legend()
    plt.tight_layout()
    return fig 

def plot_residuals_model(y_test,y_pred,name):
    fig,axes = plt.subplots(1,2,figsize=(13,4))
    residuals = np.array(y_test)-np.array(y_pred)
    axes[0].scatter(np.array(y_pred),residuals, alpha=0.3, s=8, color = 'steelblue')
    axes[0].axhline(0,color = 'red',linestyle = '--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residuals vs Predicated-{name}')
    axes[1].hist(residuals,bins = 50,color='seagreen',edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_title('Residuals Distribution')
    plt.tight_layout()
    return fig

def cross_validate_model(model,X,y,cv=5):
        tcsv = TimeSeriesSplit(n_splits=cv)
        scores = cross_val_score(model,X,y,cv=tcsv,scoring = 'r2')
        print(f"CV R2_Scores : {scores.round(4)}")
        print(f"Mean R2  : {scores.mean():.4f} (+/-{scores.std():.4f})")
        return scores

def compare_models(results_list):
        df_results = pd.DataFrame(results_list)
        return df_results.sort_values('R2',ascending=False).reset_index(drop=True)


        

    
    
    