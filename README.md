# Air quality forecasting using Transformers

This repository host the code for the experiments of my master thesis on the subject:  
*AIR QUALITY FORECASTING IN METROPOLITAN AREAS*

Data preprocessing, model development and training were performed in a Python3 environment, using *pandas* and *numpy* libraries for data manipulation, *Tensorflow* and *Keras* for building and training models and *Jupyter* notebooks for running and organizing code. This study is based on several tutorials and articles referenced at the end of the notebook.

## Problem

To what extent a *Transformer* model based on the *Attention* mechanism could outperform or at least have comparable results to the reference deep learning models (LSTM, CNN1D, CNN2D).

## The dataset

Madrid city centre air pollutant and weather parameters recorded between 2012-2017.

Main pollutants: 

   - 'PM2.5'   --> Particles smaller than 2.5 μm level measured in μg/m³. The size of these particles allow them to penetrate into the gas
                exchange regions of the lungs (alveolus) and even enter the arteries. Long-term exposure is proven to be related to low birth weight and high blood pressure in newborn babies.
   - 'PM10'    --> Particles smaller than 10 μm level measured in μg/m³. Even though the cannot penetrate the alveolus, they can still penetrate through the lungs and 
                affect other organs. Long term exposure can result in lung cancer and cardiovascular complications.
   - 'CO'      --> Carbon monoxide level measured in mg/m³. Carbon monoxide poisoning involves headaches, dizziness and confusion in short 
                exposures and can result in loss of consciousness, arrhythmias, seizures or even death in the long term.
   - 'NO2'     --> Nitrogen dioxide level measured in μg/m³. Long-term exposure is a cause of chronic lung diseases, and are harmful for the 
                vegetation.
   - 'NO'      --> Nitric oxide level measured in μg/m³. This is a highly corrosive gas generated among others by motor vehicles and fuel 
                burning processes.
   - 'O3'      --> Ozone level measured in μg/m³. High levels can produce asthma, bronchytis or other chronic pulmonary diseases in sensitive 
                groups or outdoor workers.
   - 'SO2'     --> Sulphur dioxide level measured in μg/m³. High levels of sulphur dioxide can produce irritation in the skin and membranes, 
                and worsen asthma or heart diseases in sensitive groups.
   - 'BEN'     --> Benzene level measured in μg/m³. Benzene is a eye and skin irritant, and long exposures may result in several types of 
                cancer, leukaemia and anaemias. Benzene is considered a group 1 carcinogenic to humans by the IARC.   
   - 'TOL'     --> Toluene (methylbenzene) level measured in μg/m³. Long-term exposure to this substance (present in tobacco smkoke as well) 
                can result in kidney complications or permanent brain damage.
   - 'EBE'     --> Ethylbenzene level measured in μg/m³. Long term exposure can cause hearing or kidney problems and the IARC has concluded 
                that long-term exposure can produce cancer.

Weather parameters:

   - 'TEMP'        --> Temperature measured in degrees Celsius.
   - 'PRES'        --> Atmospheric pressure measured in mb.
   - 'RH'          --> Relative humidity expressed in a percentage representing the amount of water vapor in the air at a given temperature 
                    compared to the max possible water vapor amount at that same temperature.
   - 'PRE'      --> Precipitation measured in mm, numerically equal to the number of kilograms of water per square meter.
   - 'DWP'         --> Dew point measured in degrees Celsius is the temperature the air needs to be cooled to (at constant pressure) in order 
                    to achieve a relative humidity (RH) of 100%.
   - 'WS'     --> Wind speed measured in Km/h.
   - 'WG'      --> Wind gusts speed measured in Km/h.
   - 'WD'       --> Wind direction measured in degrees.
   - 'VIS'         --> Visibility measured in km is the measure of the distance at which an object or light can be clearly discerned.
   - 'FL'          --> Feels like measured in degrees Celsius is the apparent temperature perceived by human body, based on temperature, 
                    relative humidity and wind speed.
   - 'HI'          --> Heat index measured in degrees Celsius is the apparent temperature perceived by human body, based on temperature and 
                    relative humidity.
   - 'WC'          --> Wind chill measured in degrees Celsius is the apparent temperature perceived by human body, based on temperature and 
                    wind speed.
   - 'CC'          --> Cloud cover expressed in a percentage representing the amount of sky covered by clouds.
   - 'SH'          --> Sun hour is a count representing the number of Sun-hours in a day (i.e. 1000 watts of energy shining on 1 square meter 
                    of surface for 1 hour)
   - 'UV'          --> UV index is a dimensionless measure of the strength of the sunburn-producing ultraviolet radiation at a particular place 
                    and time.
   - 'MIL'         --> Moon illumination is a percentage representing the partition of the moon illuminated by the sun.

## Models
- Transformer model adapted for regression problems (time-series forecasting)
- Transformer model with learnable parameters (Time2Vec representation of time)
- LSTM
- 1D CNN
- 2D CNN

## Evaluation metrics
- MAE
- RMSE
- NRMSE (min-max range normalization)
