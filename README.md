# Bike_Sharing_inDask

## Auhtor: Moritz Steinbrecher
=========================================
Project
=========================================
The leverages a dataset of bike sharing transactions in Washington DC.

Aim: 
  - Regression: 
		Predication of bike rental count hourly or daily based on the environmental and seasonal settings.
    
Files

	- Readme.txt
	- hour.csv : bike sharing counts aggregated on hourly basis. Records: 17379 hours
	
=========================================
Dataset characteristics
=========================================	
Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
	
	- instant: record index
	- dteday : date
	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered
    
=========================================
Dask application
=========================================
Whereas the original Analysis was assessed using Pandas and sklearn for mashine learning, this project replaces the use of such packages by leveraging the capabilities of dask and dask.ml.
  
  - a git repository and a automatic or programmatic download of the data before the analysis 
  - Use of dask.dataframe and distributed.Client for all the data manipulation
  - Use of Dask-ML for distributed training and model selection
  
