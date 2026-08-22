# Air Quality Forecasting

A beginner-level regression project that forecasts the hourly PM2.5 concentration (µg/m³) at the US Embassy in Beijing using calendar, lag, rolling-mean, and meteorological features engineered from five years of hourly air-quality and weather data (UCI Beijing PM2.5 dataset, 2010 to 2014).

### **Problem Statement**

Given the time of day, day of week, month, the recent weather (dew point, temperature, pressure, wind, snow, rain), the wind direction, and the PM2.5 values observed in the previous hour and the same hour yesterday, predict the continuous PM2.5 concentration (pm2.5) in µg/m³ for the current hour.

### **Dataset**

* Source: Beijing PM2.5 Data. UCI Machine Learning Repository
* Raw samples: 43,824 hourly readings (Jan 2010. Dec 2014)
* Missing target: 2,067 rows  have no pm2.5 reading and are dropped
* Samples after feature engineering: 22,914 rows (rows with a missing target or lag/rolling warmup NaNs removed).

### **Project Structure**

C:.
└───Air_quality_Forecasting
    │   01_eda.ipynb
    │   02_data_cleaning.ipynb
    │   model_building.ipynb
    │   Readme.md
    │   Requirement.txt
    │   utils.py
    │
    ├───data
    │       air_quality_features.csv
    │       PRSA_data.csv

Run notebooks in order: 01_eda.ipynb → 02_data_cleaning.ipynb → 03_model_building.ipynb

### **Results**

7 regressors + 1 tuned Gradient Boosting variant, evaluated on a chronological 80/20 split (train = 2010, early-2014, test = Feb 2014. Dec 2014). No shuffling, to respect time-series order.

### **Key Findings**

* Target has strong right skewness distrubution with (skew=1.85),Extensive outliers signify periodic severe air pollution spikes.
* High Volatility : Over the time (5Y) pollution remains consistant.
* Air Pollution peaks overnight,on suturdays, and during winter season, while droping to it's lowest levels mid afternoon,on mondays, during summer.
* PM2.5 concentrations decreses with high wind speed('Iws' : - 0.24) and warmer temp ('TEMP' : -0.09 ) , but increses with higher dew points ('DEWP' : 0.17), while precipitation ( 'Ir' ) and snow ( 'Is' ) showminimal impact.

* Air Pollution peaks during Calm/Variable (CV), while North-West (NW) winds bring cleanest air.

* A time-series analysis of pm2.5 over a single week in Jan 2014, reveals distinct daily cyclic variations disruped by a critical mid week pollution that peaked above with (PM2.5=700).

* The immediate past hour ( lag_1) is incredibly accurate predictor of current pollution with near perfect score of r = 0.966,while 24-hour daily avg (roll_24) provide broder, more flexible baseline with a lower but strong score of r = 0.719.

### Teck Stacks

* Jupyter
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn

### Getting Started

pip install -r requirements.txt 

jupyter notebook 01_eda.ipynb
