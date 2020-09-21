-- GEO1001.2020 --hw01
-- [ ANDRONIKI PAVLIDOU ]
-- [5267536]

# This code was made for the computation of statistical indexes, histograms, boxplots, scatterplot
    It is a .py file. 

### INPUT: 5 excel (19 variables concered weather data):  excel title: ._HEAT - "sensors name"_finals.xlsl

### MAIN BODY: functions

### OUTPUT: histograms, plots, txt file, values

locations: read excel files (path)
no_rows = [0, 1, 2, 3]: remove these rows from excel (keeps only the values)

## AFTER LESSON A1##
df_a ... df_e : pandas data frames that contain the data from excel files 

def read: reading the location and each columns required for ansewring the questions-->

temp_a...temp_e: refers to Temperature values for sensors A ... E
dir_t_a ... dir_t_e: refers to Wind direction values for sensors A ... E
wind_s_a ... wind_s_e: refers to Wind Speed values for sensors A ... E
wet_bg_a ... wet_bg_e: refers to Wet Bulb Globe values for sensors A ... E
cross_s_a ...cross_s_e: refers to Crosswind Speed values for sensors A ... E

def statistics: compute the mean, variance, stadard deviation for the 5 Sensors for 19 variables

def hist_temp(a, b, c, d): compute the histograms for the viarable of Temperature twice: 1 bins=5, 1 bins =50

def hist_temp_sens(a, b, c, d, e):compute frequency polygons for all sensors for the variable of Temperature

def boxplt(a, b, c, d, e, f, g, h, i ,j, k, l ,m, n, o): compute 3 boxplots 
    3.1 Wind Direction
    3.2 Wind Speed              for all sensors
    3.3 Temperature 

## AFTER LESSON A2##

def pmf(sample1, sample2, sample3, sample4, sample5): compute Probability Mass Function for all sensors for the variable of Temperature

def pdf_temp(sample1, sample2, sample3, sample4, sample5): compute Probability Density Function for all sensors for the variable of Temperature

def cdf_temp(sample1, sample2, sample3, sample4, sample5): compute Cumulative Density Function for all sensors for the variable of Temperature

def kernel_density(sample1, sample2, sample3, sample4, sample5): compute Kernel Density Estimation for all sensors for the variable of Wind Speed
## NOTE: in KDE something in labels does not work right. Instead of the black line, in legend, the grey diagram is showned as KDE. 

## AFTER LESSON A3##

def correlation(a, b, c, d, e, f): compute the correlations between sensors for 3 variables 
    3.1 Temperature
    3.2 Wet Bulb Globe
    3.3 Crosswind Speed
In this function followed these steps: 
1.interpolate to equal size samples (the excel data do not have the same sizes)
2.normalize because variables have different units
3.normalize sensors's values (optional)

excel_data: Is a list with all the columns of excels

def cdf_wdp(sample1, sample2, sample3, sample4, sample5): compute Cumulative Density Function --> NOTE: cdf function existed already but since its specific format (Temperature) I created a new one --> wind speed 

## AFTER LESSON A4##

def sensors(a): compute 95% confidence intervals -->  

file = open('Confidence results.txt', 'a')
file.write("Confidence Intervals :" + str(start) + " , " + str(end) + "\n") :::: This is the code used for .txt file creation
file.close()

def student_t(arr1, arr2): compute probability values for 4 combinations of sensors--> returns t, p -values which concern the variables of Wind Speed and Temperature

### NOTE: txt file commands are in "#comment" in order not to add the same values every time I run the program. 

## Bonus question:
def function(a): compute the mean for the 35 days for each sensor

all_s : table with all mean/sensor


## ΗΟW IT WORKS 
When you run the code --> for next plot close the previous one
Possible warnings for python libraries shown in terminal do not affect the program runs correctly. 

In the uploaded folder there are:
1. geo1001_hw01.py
2. hw01_PAVLIDOU_ANDRONIKI.tex
3. reference1.bib
4. README.md
5. Confidence txt file (the first 5 values concern WIND SPEED, the othe 5 concern TEMPERATURE)
6. hw01 <--data
