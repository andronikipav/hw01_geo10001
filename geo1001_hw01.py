# -- GEO1001.2020 --hw01
# -- [ ANDRONIKI PAVLIDOU ]
#-- [5267536]


import numpy as np
import statistics as stat
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import scipy
from scipy import stats
import pandas as pd
import seaborn as sns
from scipy.stats import sem, t
from datetime import datetime,timedelta
#from matplotlib import pyplot as plt


################################################# DATA####################################################

locations = ["hw01/HEAT - A_final.xls", "hw01/HEAT - B_final.xls", "hw01/HEAT - C_final.xls", "hw01/HEAT - D_final.xls", "hw01/HEAT - E_final.xls"]
no_rows = [0, 1, 2, 3]

df_a = pd.read_excel("hw01/HEAT - A_final.xls", nrows=None, header=3)
df_b = pd.read_excel("hw01/HEAT - B_final.xls", nrows=None, header=3)
df_c = pd.read_excel("hw01/HEAT - C_final.xls", nrows=None, header=3)
df_d = pd.read_excel("hw01/HEAT - D_final.xls", nrows=None, header=3)
df_e = pd.read_excel("hw01/HEAT - E_final.xls", nrows=None, header=3)

################### Data Frame ############################################

df_a = df_a.drop([0]).reset_index(drop=True)
df_b = df_b.drop([0]).reset_index(drop=True)
df_c = df_c.drop([0]).reset_index(drop=True)
df_d = df_d.drop([0]).reset_index(drop=True)
df_e = df_e.drop([0]).reset_index(drop=True)


def read(loc, col):

    return pd.read_excel(locations[loc], nrows=None,)

temp_a = df_a["Temperature"]
temp_b = df_b["Temperature"]
temp_c = df_c["Temperature"]
temp_d = df_d["Temperature"]
temp_e = df_e["Temperature"]

dir_t_a = df_a["Direction ‚ True"]
dir_t_b = df_b["Direction ‚ True"]
dir_t_c = df_c["Direction ‚ True"]
dir_t_d = df_d["Direction ‚ True"]
dir_t_e = df_e["Direction ‚ True"]

wind_s_a = df_a["Wind Speed"]
wind_s_b = df_b["Wind Speed"]
wind_s_c = df_c["Wind Speed"]
wind_s_d = df_d["Wind Speed"]
wind_s_e = df_e["Wind Speed"]

wet_bg_a = df_a["WBGT"]
wet_bg_b = df_b["WBGT"]
wet_bg_c = df_c["WBGT"]
wet_bg_d = df_d["WBGT"]
wet_bg_e = df_e["WBGT"]

cross_s_a = df_a["Crosswind Speed"]
cross_s_b = df_b["Crosswind Speed"]
cross_s_c = df_c["Crosswind Speed"]
cross_s_d = df_d["Crosswind Speed"]
cross_s_e = df_e["Crosswind Speed"]

######################################## AFTER LESSON A1 ######################################################

def statistics(loc, col):

    a = pd.read_excel(loc, skiprows=no_rows, usecols=col)
    return (str(a.values.mean()), str(a.values.var()), str(a.values.std()))

for i in range(5):  
        dir_t = statistics(locations[i], [1])
        wind_s = statistics(locations[i], [2])
        c_wind_s = statistics(locations[i], [3])
        h_wind_s = statistics(locations[i], [4])
        temp = statistics(locations[i], [5])
        gl_temp = statistics(locations[i], [6])
        wind_ch = statistics(locations[i], [7])
        rel_humid = statistics(locations[i], [8])
        heat_str = statistics(locations[i], [9])
        dew_p = statistics(locations[i], [10])
        psy_wet = statistics(locations[i], [11])
        stat_press = statistics(locations[i], [12])
        barom_press = statistics(locations[i], [13])
        alt = statistics(locations[i], [14])
        dens_alt = statistics(locations[i], [15])
        na_wet = statistics(locations[i], [16])
        wbgt = statistics(locations[i], [17])
        twl = statistics(locations[i], [18])
        dir_mag = statistics(locations[i], [19])

        print("\n", "=== VALUES FOR SENSOR:", i+1, "=================================================="*2, "\n" +
          "True Direction-> \t mean:", dir_t[0], "\t", "variance:", dir_t[1], "\t", "standard deviation:", dir_t[2], "\n"
          + "Wind Speed-> \t\t mean:", wind_s[0], "\t", "variance:", wind_s[1], "\t", "standard deviation:", wind_s[2], "\n"
          + "Crosswind Speed-> \t mean:", c_wind_s[0], "\t", "variance:", c_wind_s[1], "\t", "standard deviation:", c_wind_s[2], "\n"
          + "Headwind Speed-> \t mean:", h_wind_s[0], "\t", "variance:", h_wind_s[1], "\t", "standard deviation:", h_wind_s[2], "\n"
          + "Temperature-> \t\t mean:", temp[0], "\t", "variance:", temp[1], "\t", "standard deviation:", temp[2], "\n"
          + "Globe Temperature-> \t mean:", gl_temp[0], "\t", "variance:", gl_temp[1], "\t", "standard deviation:", gl_temp[2], "\n"
          + "Wind chill-> \t\t mean:", wind_ch[0], "\t", "variance:", wind_ch[1], "\t", "standard deviation:", wind_ch[2], "\n"
          + "Relative humidity-> \t mean:", rel_humid[0], "\t", "variance:", rel_humid[1], "\t", "standard deviation:", rel_humid[2], "\n"
          + "Heat stress index-> \t mean:", heat_str[0], "\t", "variance:", heat_str[1], "\t", "standard deviation:", heat_str[2], "\n"
          + "Dew point-> \t\t mean:", dew_p[0], "\t", "variance:", dew_p[1], "\t", "standard deviation:", dew_p[2], "\n"
          + "Psychro Wet Bulb Temp->  mean:", psy_wet[0], "\t", "variance:", psy_wet[1], "\t", "standard deviation:", psy_wet[2], "\n"
          + "Station pressure-> \t mean:", stat_press[0], "\t", "variance:", stat_press[1], "\t", "standard deviation:", stat_press[2], "\n"
          + "Barometric pressure-> \t mean:", barom_press[0], "\t", "variance:", barom_press[1], "\t", "standard deviation:", barom_press[2], "\n"
          + "Altitude-> \t\t mean:", alt[0], "\t", "variance:", alt[1], "\t", "standard deviation:", alt[2], "\n"
          + "Density Altitude-> \t mean:", dens_alt[0], "\t", "variance:", dens_alt[1], "\t", "standard deviation:", dens_alt[2], "\n"
          + "NA Wet Bulb Temperature->mean:", na_wet[0], "\t", "variance:", na_wet[1], "\t", "standard deviation:", na_wet[2], "\n"
          + "WBGT-> \t\t\t mean:", wbgt[0], "\t", "variance:", wbgt[1], "\t", "standard deviation:", wbgt[2], "\n"
          + "TWL-> \t\t\t mean:", twl[0], "\t", "variance:", twl[1], "\t", "standard deviation:", twl[2], "\n"
          + "Direction ‚ Mag-> \t mean:", dir_mag[0], "\t", "variance:", dir_mag[1], "\t", "standard deviation:", dir_mag[2])

def hist_temp(a, b, c, d):

    fig, axes = plt.subplots(1, 2)

    axes[0].set_ylabel("Frequency")
    axes[1].set_ylabel("Frequency")
    axes[0].set_xlabel("Temperature")
    axes[1].set_xlabel("Temperature")
    a.hist(ax=axes[0], bins=b, color="c")  # plt.hist(x="5 sensors", bins)
    a.hist(ax=axes[1], bins=c, color="c")

    plt.grid(axis='y', alpha=0.85)
    plt.suptitle('Histogram of Temperature values Sensor '+d)
    plt.show()


hist_temp(temp_a.astype(float), 5, 50, 'A')
hist_temp(temp_b.astype(float), 5, 50, 'B')
hist_temp(temp_c.astype(float), 5, 50, 'C')
hist_temp(temp_d.astype(float), 5, 50, 'D')
hist_temp(temp_e.astype(float), 5, 50, 'E')


def hist_temp_sens(a, b, c, d, e):

    fs = 9

    fig = plt.figure(figsize=(21, 6))
    ax_a = fig.add_subplot(111)
    ax_b = fig.add_subplot(111)
    ax_c = fig.add_subplot(111)
    ax_d = fig.add_subplot(111)
    ax_e = fig.add_subplot(111)

    [fr_a, bins] = np.histogram(a, bins=20)  # plt.hist(x="5 sensors", bins)
    [fr_b, bins] = np.histogram(b, bins=20)
    [fr_c, bins] = np.histogram(c, bins=20)
    [fr_d, bins] = np.histogram(d, bins=20)
    [fr_e, bins] = np.histogram(e, bins=20)

    cdf_A = np.cumsum(fr_a)
    cdf_B = np.cumsum(fr_b)
    cdf_C = np.cumsum(fr_c)
    cdf_D = np.cumsum(fr_d)
    cdf_E = np.cumsum(fr_e)

    # Create 1 plot where frequency poligons for the 5 sensors Temperature values overlap in different colors with a legend.

    x = np.linspace(0, 10, 1000)
    ax_a.plot(bins[:-1], cdf_A, label="Sensor A")
    ax_b.plot(bins[:-1], cdf_B, label="Sensor b")
    ax_c.plot(bins[:-1], cdf_C, label="Sensor C")
    ax_d.plot(bins[:-1], cdf_D, label="Sensor D")
    ax_e.plot(bins[:-1], cdf_E, label="Sensor E")

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Temperature in °C', fontsize=fs)
    plt.ylabel('Cumulative number of samples', fontsize=fs)
    plt.title("Sensors's frequency polygons")
    plt.legend(prop={"size": 10}, title="Legend")
    plt.tick_params(labelsize=fs)

    plt.show()


hist_temp_sens(temp_a.astype(float), temp_b.astype(float), temp_c.astype(float), temp_d.astype(float), temp_e.astype(float))

# Generate 3 plots that include the 5 sensors boxplot for: Wind Speed, Wind Direction and Temperature.


def boxplt(a, b, c, d, e, f, g, h, i ,j, k, l ,m, n, o):

    fig, axes = plt.subplots(1,3)

    values1 = [a, b, c, d, e]
    values2= [f, g, h, i, j]
    values3 = [k, l, m, n, o]

    axes[0].boxplot(values1, showmeans=True)
    axes[0].set_ylabel("Wind Speed")
    axes[0].set_xlabel("Sensors")
    axes[0].set_title("Boxplot for Wind Speed")

    axes[1].boxplot(values2, showmeans=True)
    axes[1].set_ylabel("Wind Direction")
    axes[1].set_xlabel("Sensors")
    axes[1].set_title("Boxplot for Wind Direction")

    axes[2].boxplot(values3, showmeans=True)
    axes[2].set_ylabel("Temperature")
    axes[2].set_xlabel("Sensors")
    axes[2].set_title("Boxplot for Temperature")

    plt.show()


boxplt(wind_s_a.astype(float), wind_s_b.astype(float), wind_s_c.astype(float), wind_s_d.astype(float), wind_s_e.astype(float), 
dir_t_a.astype(float), dir_t_b.astype(float), dir_t_c.astype(float), dir_t_d.astype(float), dir_t_e.astype(float), temp_a.astype(float), 
temp_b.astype(float), temp_c.astype(float), temp_d.astype(float), temp_e.astype(float))


######################################## AFTER LESSON A2 ######################################################

def pmf(sample1, sample2, sample3, sample4, sample5):
    c1 = sample1.value_counts()
    p1 = c1/len(sample1)

    c2 = sample2.value_counts()
    p2 = c2/len(sample2)

    c3 = sample3.value_counts()
    p3 = c3/len(sample3)

    c4 = sample4.value_counts()
    p4 = c4/len(sample4)

    c5 = sample5.value_counts()
    p5 = c5/len(sample5)

######################## Probability Mass Functions ##################################
    df1 = p1
    df2 = p2
    df3 = p3
    df4 = p4
    df5 = p5

    fs = 9

    fig = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    fig.suptitle("PMF Temperature")

    c1 = df1.sort_index()
    ax1 = fig.add_subplot(231)
    ax1.bar(c1.index, c1, width=0.1)
    plt.title("PMF of temperature: Sensor A")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(" PMF values ")

    c2 = df2.sort_index()
    ax2 = fig.add_subplot(232)
    ax2.bar(c2.index, c2, width=0.1)
    plt.title("PMF of temperature: Sensor B")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(" PMF values ")

    c3 = df3.sort_index()
    ax3 = fig.add_subplot(233)
    ax3.bar(c3.index, c3, width=0.1)
    plt.title("PMF of temperature: Sensor C")
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel(" PMF values ")

    c4 = df4.sort_index()
    ax4 = fig.add_subplot(223)
    ax4.bar(c4.index, c4, width=0.1)
    plt.title("PMF of temperature: Sensor D")
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel(" PMF values ")

    c5 = df5.sort_index()
    ax5 = fig.add_subplot(224)
    ax5.bar(c5.index, c5, width=0.1)
    plt.title("PMF of temperature: Sensor E")
    ax5.set_xlabel("Temperature")
    ax5.set_ylabel(" PMF values ")

    plt.tick_params(labelsize=fs)

    plt.show()


pmf(temp_a.astype(float), temp_b.astype(float),(temp_c.astype(float)), temp_d.astype(float), temp_e.astype(float))

######################## Probability Density Functions ##################################


def pdf_temp(sample1, sample2, sample3, sample4, sample5):

    fig = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    fig.suptitle("PDF Temperature")

    ax1 = fig.add_subplot(231)
    a1 = ax1.hist(x=sample1, bins=30, density=True,color='b', alpha=0.7, rwidth=0.1)
    plt.title("PDF of temperature: Sensor A")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(" PDF values ")

    ax2 = fig.add_subplot(232)
    a2 = ax2.hist(x=sample2, bins=30, density=True,color='b', alpha=0.7, rwidth=0.1)
    plt.title("PDF of temperature: Sensor B")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(" PDF values ")

    ax3 = fig.add_subplot(233)
    a3 = ax3.hist(x=sample3, bins=30, density=True,color='b', alpha=0.7, rwidth=0.1)
    plt.title("PDF of temperature: Sensor C")
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel(" PDF values ")

    ax4 = fig.add_subplot(223)
    a4 = ax4.hist(x=sample4, bins=30, density=True,color='b', alpha=0.7, rwidth=0.1)
    plt.title("PDF of temperature: Sensor D")
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel(" PDF values ")

    ax5 = fig.add_subplot(224)
    a5 = ax5.hist(x=sample5, bins=30, density=True,color='b', alpha=0.7, rwidth=0.1)
    plt.title("PDF of temperature: Sensor E")
    ax5.set_xlabel("Temperature")
    ax5.set_ylabel(" PDF values ")

    plt.show()


pdf_temp(temp_a.astype(float), temp_b.astype(float), temp_c.astype(float), temp_d.astype(float), temp_e.astype(float))

######################## Cumulative Density Functions ##################################


def cdf_temp(sample1, sample2, sample3, sample4, sample5):

    fig = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    fig.suptitle("CDF Temperature")

    ax1 = fig.add_subplot(231)
    a1 = ax1.hist(x=sample1, bins=30, cumulative=True, density=True, histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2, a1[0], color='k')
    ax1.set_title("CDF of Temperature: Sensor A")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(" CDF values ")

    ax2 = fig.add_subplot(232)
    a2 = ax2.hist(x=sample2, bins=30, cumulative=True, density=True,histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2, a2[0], color='k')
    ax2.set_title("CDF of Temperature: Sensor B")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(" CDF values ")

    ax3 = fig.add_subplot(233)
    a3 = ax3.hist(x=sample3, bins=27, cumulative=True, density=True,histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2, a3[0], color='k')
    ax3.set_title("CDF of Temperature: Sensor B")
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel(" CDF values ")

    ax4 = fig.add_subplot(223)
    a4 = ax4.hist(x=sample4, bins=30, cumulative=True, density=True,histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2, a4[0], color='k')
    ax4.set_title("CDF of Temperature: Sensor B")
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel(" CDF values ")

    ax5 = fig.add_subplot(224)
    a5 = ax5.hist(x=sample5, bins=30, cumulative=True, density=True, histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2, a5[0], color='k')
    ax5.set_title("CDF of Temperature: Sensor E")
    ax5.set_xlabel("Temperature")
    ax5.set_ylabel(" CDF values ")

    plt.show()


cdf_temp(temp_a.astype(float), temp_b.astype(float), temp_c.astype(float), temp_d.astype(float), temp_e.astype(float))

####################### Kernel  Density - Wind Speed Values #############################


def kernel_density(sample1, sample2, sample3, sample4, sample5):

    fig = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    fig.suptitle("Kernel Density")

    ax1 = fig.add_subplot(231)
    a1 = ax1.hist(x=sample1, bins=27, density=True, color='b',alpha=0.7, rwidth=0.1, label="PDF")
    sns.distplot(sample1, color='k', ax=ax1, hist=True, kde=True, label="KDE")
    plt.title("Kernel Density - Sensor A")
    ax1.set_xlabel("Wind Speed")
    ax1.set_ylabel(" KDE values ")

    ax2 = fig.add_subplot(232)
    a2 = ax2.hist(x=sample2, bins=27, density=True, color='b',alpha=0.7, rwidth=0.1, label="PDF")
    sns.distplot(sample1, color='k', ax=ax2, hist=True, kde=True, label="KDE")

    plt.title("Kernel Density - Sensor B")
    ax2.set_xlabel("Wind Speed")
    ax2.set_ylabel(" KDE values ")

    ax3 = fig.add_subplot(233)
    a3 = ax3.hist(x=sample3, bins=27, density=True, color='b',alpha=0.7, rwidth=0.1, label="PDF")
    sns.distplot(sample3, color='k', ax=ax3, hist=True, kde=True, label="KDE")
    plt.title("Kernel Density - Sensor C")
    ax3.set_xlabel("Wind Speed")
    ax3.set_ylabel(" KDE values ")

    ax4 = fig.add_subplot(223)
    a4 = ax4.hist(x=sample4, bins=27, density=True, color='b',alpha=0.7, rwidth=0.1, label="PDF")
    sns.distplot(sample4, color='k', ax=ax4, hist=True, kde=True, label="KDE")
    plt.title("Kernel Density - Sensor D")
    ax4.set_xlabel("Wind Speed")
    ax4.set_ylabel(" KDE values ")

    ax5 = fig.add_subplot(224)
    a5 = ax5.hist(x=sample5, bins=27, density=True, color='b',alpha=0.7, rwidth=0.1, label="PDF")
    sns.distplot(sample5, color='k', ax=ax5, hist=True, kde=True, label="KDE")
    plt.title("Kernel Density - Sensor E")
    ax5.set_xlabel("Wind Speed")
    ax5.set_ylabel(" KDE values ")
    plt.legend()

    plt.show()


kernel_density(wind_s_a.astype(float), wind_s_b.astype(float), wind_s_c.astype(float), wind_s_d.astype(float), wind_s_e.astype(float))


################################# AFTER LECTURE A3 ######################################
def correlation(a, b, c, d, e, f):

    # Step 1 --- interpolate to equal size samples
    ab = np.interp(np.linspace(0, len(b), len(b)),np.linspace(0, len(a), len(a)), a)
    ac = np.interp(np.linspace(0, len(c), len(c)),np.linspace(0, len(a), len(a)), a)
    ad = np.interp(np.linspace(0, len(d), len(d)), np.linspace(0, len(a), len(a)), a)
    ae = np.interp(np.linspace(0, len(e), len(e)),np.linspace(0, len(a), len(a)), a)

    bc = np.interp(np.linspace(0, len(c), len(c)),np.linspace(0, len(b), len(b)), b)
    bd = np.interp(np.linspace(0, len(d), len(d)),np.linspace(0, len(b), len(b)), b)
    be = np.interp(np.linspace(0, len(e), len(e)),np.linspace(0, len(b), len(b)), b)

    cd = np.interp(np.linspace(0, len(d), len(d)),np.linspace(0, len(c), len(c)), c)
    ce = np.interp(np.linspace(0, len(e), len(e)),np.linspace(0, len(c), len(c)), c)

    de = np.interp(np.linspace(0, len(e), len(e)),np.linspace(0, len(d), len(d)), d)

    # Step 2 --- normalize because variables have different units
    norm_ab = (ab - ab.mean())/ab.std()
    norm_ac = (ac - ac.mean())/ac.std()
    norm_ad = (ad - ad.mean())/ad.std()
    norm_ae = (ae - ae.mean())/ae.std()

    norm_bc = (bc - bc.mean())/bc.std()
    norm_bd = (bd - bd.mean())/bd.std()
    norm_be = (be - be.mean())/be.std()

    norm_cd = (cd - cd.mean())/cd.std()
    norm_ce = (ce - ce.mean())/ce.std()

    norm_de = (de - de.mean())/de.std()

 # normalize sensors's values

    anorm = (a - a.mean()/a.std())
    bnorm = (b - b.mean()/b.std())
    cnorm = (c - c.mean()/c.std())
    dnorm = (d - d.mean()/d.std())
    enorm = (e - e.mean()/e.std())

    pearson = []
    spearman = []

    ab_pcoef = stats.pearsonr(norm_ab, bnorm)[0]
    ab_prcoef = stats.spearmanr(norm_ab, bnorm)[0]
    pearson.append(ab_pcoef)
    spearman.append(ab_prcoef)

    ac_pcoef = stats.pearsonr(norm_ac, cnorm)[0]
    ac_prcoef = stats.spearmanr(norm_ac, cnorm)[0]
    pearson.append(ac_pcoef)
    spearman.append(ac_prcoef)

    ad_pcoef = stats.pearsonr(norm_ad, dnorm)[0]
    ad_prcoef = stats.spearmanr(norm_ad, dnorm)[0]
    pearson.append(ad_pcoef)
    spearman.append(ad_prcoef)

    ae_pcoef = stats.pearsonr(norm_ae, enorm)[0]
    ae_prcoef = stats.spearmanr(norm_ae, enorm)[0]
    pearson.append(ae_pcoef)
    spearman.append(ae_prcoef)

    bc_pcoef = stats.pearsonr(norm_bc, cnorm)[0]
    bc_prcoef = stats.spearmanr(norm_bc, cnorm)[0]
    pearson.append(bc_pcoef)
    spearman.append(bc_prcoef)

    bd_pcoef = stats.pearsonr(norm_bd, dnorm)[0]
    bd_prcoef = stats.spearmanr(norm_bd, dnorm)[0]
    pearson.append(bd_pcoef)
    spearman.append(bd_prcoef)

    be_pcoef = stats.pearsonr(norm_be, enorm)[0]
    be_prcoef = stats.spearmanr(norm_be, enorm)[0]
    pearson.append(be_pcoef)
    spearman.append(be_prcoef)

    cd_pcoef = stats.pearsonr(norm_cd, dnorm)[0]
    cd_prcoef = stats.spearmanr(norm_cd, dnorm)[0]
    pearson.append(cd_pcoef)
    spearman.append(cd_prcoef)

    ce_pcoef = stats.pearsonr(norm_ce, enorm)[0]
    ce_prcoef = stats.spearmanr(norm_ce, enorm)[0]
    pearson.append(ce_pcoef)
    spearman.append(ce_prcoef)

    de_pcoef = stats.pearsonr(norm_de, enorm)[0]
    de_prcoef = stats.spearmanr(norm_de, enorm)[0]
    pearson.append(de_pcoef)
    spearman.append(de_prcoef)

    xlabel = ["AB", "AC", "AD", "AE", "BC", "BD", "BE", "CD", "CE", "DE"]

    fig = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    fig.suptitle("Correlations :" + f)

    ax1 = fig.add_subplot(121)
    ax1.scatter(xlabel, pearson)
    ax1.set_xlabel('Sensors combination')
    ax1.set_ylabel('Pearson Correlations')

    ax2 = fig.add_subplot(122)
    ax2.scatter(xlabel, spearman)
    ax2.set_xlabel('Sensors combination')
    ax2.set_ylabel('Spearman Correlations')

    plt.show()


correlation(temp_a.astype(float), temp_b.astype(float), temp_c.astype(float), temp_d.astype(float), temp_e.astype(float), "Temperature")
correlation(wet_bg_a.astype(float), wet_bg_b.astype(float), wet_bg_c.astype(float), wet_bg_d.astype(float), wet_bg_e.astype(float), "Wet Bulb Globe")
correlation(cross_s_a.astype(float), cross_s_b.astype(float), cross_s_c.astype(float), cross_s_d.astype(float), cross_s_e.astype(float), "Crosswind")

############################### AFTER LECTURE 4#######################################
excel_data = ["Direction ‚ True", "Wind Speed", "Crosswind Speed", "Headwind Speed", "Temperature", "Globe Temperature",
              "Wind Chill", "Relative Humidity", "Heat Stress Index", "Dew Point", "Psychro Wet Bulb Temperature", "Station Pressure",
              "Barometric Pressure", "Altitude", "Density Altitude", "NA Wet Bulb Temperature", "WBGT", "TWL", "Direction ‚ Mag"]


def cdf_wdp(sample1, sample2, sample3, sample4, sample5):

    fig = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    fig.suptitle("CDF Wind Speed")

    ax1 = fig.add_subplot(231)
    a1 = ax1.hist(x=sample1, bins=30, cumulative=True, density=True,histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2, a1[0], color='k')
    ax1.set_title("CDF of Wind Speed: Sensor A")
    ax1.set_xlabel("Wind Speed")
    ax1.set_ylabel(" CDF values ")

    ax2 = fig.add_subplot(232)
    a2 = ax2.hist(x=sample2, bins=30, cumulative=True, density=True,histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2, a2[0], color='k')
    ax2.set_title("CDF of Wind Speed: Sensor A")
    ax2.set_xlabel("Wind Speed")
    ax2.set_ylabel(" CDF values ")

    ax3 = fig.add_subplot(233)
    a3 = ax3.hist(x=sample3, bins=30, cumulative=True, density=True, histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2, a3[0], color='k')
    ax3.set_title("CDF of Wind Speed: Sensor A")
    ax3.set_xlabel("Wind Speed")
    ax3.set_ylabel(" CDF values ")

    ax4 = fig.add_subplot(223)
    a4 = ax4.hist(x=sample4, bins=30, cumulative=True, density=True,histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2, a4[0], color='k')
    ax4.set_title("CDF of Wind Speed: Sensor A")
    ax4.set_xlabel("Wind Speed")
    ax4.set_ylabel(" CDF values ")

    ax5 = fig.add_subplot(224)
    a5 = ax5.hist(x=sample5, bins=30, cumulative=True, density=True,histtype="step", color='b', alpha=0.7, rwidth=0.1)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2, a5[0], color='k')
    ax5.set_title("CDF of Wind Speed: Sensor A")
    ax5.set_xlabel("Wind Speed")
    ax5.set_ylabel(" CDF values ")

    plt.show()


cdf_temp(df_a[excel_data[1]].astype(float),df_b[excel_data[1]].astype(float),df_c[excel_data[1]].astype(float),df_d[excel_data[1]].astype(float),df_e[excel_data[1]].astype(float))
cdf_wdp(df_a[excel_data[1]].astype(float), df_b[excel_data[1]].astype(float), df_c[excel_data[1]].astype(float), df_d[excel_data[1]].astype(float), df_e[excel_data[1]].astype(float))


def sensors(a):
    confidence = 0.95

    data = a
    n = len(data)
    m = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h

    #################################################  TXT FILE ############################

   # file = open('Confidence results.txt', 'a')

   # file.write("Confidence Intervals :" + str(start) + " , " + str(end) + "\n")

   # file.close()


sensors(df_a[excel_data[1]].astype(float))
sensors(df_b[excel_data[1]].astype(float))
sensors(df_c[excel_data[1]].astype(float))
sensors(df_d[excel_data[1]].astype(float))
sensors(df_e[excel_data[1]].astype(float))

sensors(df_a[excel_data[4]].astype(float))
sensors(df_b[excel_data[4]].astype(float))
sensors(df_c[excel_data[4]].astype(float))
sensors(df_d[excel_data[4]].astype(float))
sensors(df_e[excel_data[4]].astype(float))

##### Computation of p-value #####

def student_t(arr1, arr2):

    data = arr1, arr2
    t, p = stats.ttest_ind(data[0], data[1])
    print("The statistic index t is =" +str(t),"The probability value is =" +str(p))


student_t(temp_e.astype(float).values, temp_d.astype(float).values)
student_t(temp_d.astype(float).values, temp_c.astype(float).values)
student_t(temp_c.astype(float).values, temp_b.astype(float).values)
student_t(temp_b.astype(float).values, temp_a.astype(float).values)

student_t(wind_s_e.astype(float).values, wind_s_d.astype(float).values)
student_t(wind_s_d.astype(float).values, wind_s_c.astype(float).values)
student_t(wind_s_c.astype(float).values, wind_s_b.astype(float).values)
student_t(wind_s_b.astype(float).values, wind_s_a.astype(float).values)


######################################## BONUS QUESTION #######################################################



new_temp_a=df_a[["FORMATTED DATE-TIME","Temperature"]]
new_temp_b=df_b[["FORMATTED DATE-TIME","Temperature"]]
new_temp_c=df_c[["FORMATTED DATE-TIME","Temperature"]]
new_temp_d=df_d[["FORMATTED DATE-TIME","Temperature"]]
new_temp_e=df_e[["FORMATTED DATE-TIME","Temperature"]]


def function(a):
    table=pd.DataFrame()
    date="2020-06-10"

    for i in range(35):
    
        format_g=a[a["FORMATTED DATE-TIME"].astype(str).str.contains(date)]
        format_v=format_g["Temperature"].astype(float)
        table=table.append({"Days":date , "Temperature" : format_v.mean()},ignore_index=True)
        date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    return table

mean_values1 = function(new_temp_a)
mean_values2 = function(new_temp_b)
mean_values3 = function(new_temp_c)
mean_values4 = function(new_temp_d)
mean_values5 = function(new_temp_e)


mean_values1 = mean_values1.rename(columns={"Temperature":"Temperature SA"}, inplace=False)
mean_values2 = mean_values2.rename(columns={"Temperature":"Temperature SB"}, inplace=False)
mean_values3 = mean_values3.rename(columns={"Temperature":"Temperature SC"}, inplace=False)
mean_values4 = mean_values4.rename(columns={"Temperature":"Temperature SD"}, inplace=False)
mean_values5 = mean_values5.rename(columns={"Temperature":"Temperature SE"}, inplace=False)

ss1 = mean_values1["Temperature SA"]
ss2 = mean_values2["Temperature SB"]
ss3 = mean_values3["Temperature SC"]
ss4 = mean_values4["Temperature SD"]
ss5 = mean_values5["Temperature SE"]

all_s = mean_values1
all_s = all_s.join(ss2)
all_s = all_s.join(ss3)
all_s = all_s.join(ss4)
all_s = all_s.join(ss5)


max_s1 = all_s["Temperature SA"].max()
print("The temperature of the hotter day of the measurement time series is :" , mean_values1[mean_values1["Temperature SA"] == all_s["Temperature SA"].max()])

max_s2 = all_s["Temperature SB"].max()
print("The temperature of the hotter day of the measurement time series is :" , mean_values2[mean_values2["Temperature SB"] == all_s["Temperature SB"].max()])

max_s3 = all_s["Temperature SC"].max()
print("The temperature of the hotter day of the measurement time series is :" , mean_values3[mean_values3["Temperature SC"] == all_s["Temperature SC"].max()])

max_s4 = all_s["Temperature SD"].max()
print("The temperature of the hotter day of the measurement time series is :" , mean_values4[mean_values4["Temperature SD"] == all_s["Temperature SD"].max()])

max_s5 = all_s["Temperature SE"].max()
print("The temperature of the hotter day of the measurement time series is :" , mean_values5[mean_values5["Temperature SE"] == all_s["Temperature SE"].max()])


min_s1 = all_s["Temperature SA"].min()
print("The temperature of the coolest day of the measurement time series is :" , mean_values1[mean_values1["Temperature SA"] == all_s["Temperature SA"].min()])

min_s2 = all_s["Temperature SB"].min()
print("The temperature of the coolest day of the measurement time series is :" , mean_values2[mean_values2["Temperature SB"] == all_s["Temperature SB"].min()])

min_s3 = all_s["Temperature SC"].min()
print("The temperature of the coolest day of the measurement time series is :" , mean_values3[mean_values3["Temperature SC"] == all_s["Temperature SC"].min()])

min_s4 = all_s["Temperature SD"].min()
print("The temperature of the coolest day of the measurement time series is :" , mean_values4[mean_values4["Temperature SD"] == all_s["Temperature SD"].min()])

min_s5 = all_s["Temperature SE"].min()
print("The temperature of the coolest day of the measurement time series is :" , mean_values5[mean_values5["Temperature SE"] == all_s["Temperature SE"].min()])
