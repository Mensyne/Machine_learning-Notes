# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:51:19 2018

@author: tongshai
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
os.chdir(r'E:\company_file\learning\Meachine_examples\meachinelearning\line_regression\data')


if __name__ == "__main__":

    df = pd.read_csv('challenge_dataset.txt', names=['X','Y'])
    sns.regplot(x='X',y='Y',data=df,fit_reg=False)
    plt.show()
    
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = np.asarray(train_test_split(df['X'],df['Y'], test_size=0.1))
    
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x_train.values.reshape(-1,1),y_train.values.reshape(-1,1))
    print('Score:',reg.score(x_test.values.reshape(-1,1),y_test.values.reshape(-1,1)))
    
    # visualize
    x_line  = np.arange(5,25).reshape(-1,1)
    sns.regplot(x=df['X'],y=df['Y'],data=df,fit_reg=False)
    plt.plot(x_line,reg.predict(x_line))
    plt.show()
    
    #Data preprocessing
    co2_df = pd.read_csv('global_co2.csv')
    temp_df = pd.read_csv('annual_temp.csv')
    
    
    # Clean data
    co2_df = co2_df.ix[:,:2]                     # Keep only total CO2
    co2_df = co2_df.ix[co2_df['Year'] >= 1960]   # Keep only 1960 - 2010
    co2_df.columns=['Year','CO2']                # Rename columns
    co2_df = co2_df.reset_index(drop=True)                # Reset index
    
    temp_df = temp_df[temp_df.Source != 'GISTEMP']                              # Keep only one source
    temp_df.drop('Source', inplace=True, axis=1)                                # Drop name of source
    temp_df = temp_df.reindex(index=temp_df.index[::-1])                        # Reset index
    temp_df = temp_df.ix[temp_df['Year'] >= 1960].ix[temp_df['Year'] <= 2010]   # Keep only 1960 - 2010
    temp_df.columns=['Year','Temperature']                                      # Rename columns
    temp_df = temp_df.reset_index(drop=True)                                             # Reset index
    
    
    # Concatenate
    climate_change_df = pd.concat([co2_df, temp_df.Temperature], axis=1)
    print(climate_change_df.head())
    
    
    # 3D Plot
    
    fig = plt.figure()
    fig.set_size_inches(12.5, 7.5)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])
    
    ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
    ax.view_init(10, -45)
    
    #Projected 2D plots
    
    f, axarr = plt.subplots(2, sharex=True)
    f.set_size_inches(12.5, 7.5)
    axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
    axarr[0].set_ylabel('CO2 Emissions')
    axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
    axarr[1].set_xlabel('Year')
    axarr[1].set_ylabel('Relative temperature')
    
    
    
    ###
    X = climate_change_df.as_matrix(['Year'])
    Y = climate_change_df.as_matrix(['CO2', 'Temperature']).astype('float32')
    X_train, X_test, y_train, y_test = np.asarray(train_test_split(X, Y, test_size=0.1))
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print('Score: ', reg.score(X_test.reshape(-1, 1), y_test))
    
    x_line = np.arange(1960,2011).reshape(-1,1)
    p = reg.predict(x_line).T
    
    fig2 = plt.figure()
    fig2.set_size_inches(12.5, 7.5)
    ax = fig2.add_subplot(111, projection='3d')
    ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])
    ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
    ax.plot(xs=x_line, ys=p[1], zs=p[0], color='green')
    ax.view_init(10, -45)
    
    f, axarr = plt.subplots(2, sharex=True)
    f.set_size_inches(12.5, 7.5)
    axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
    axarr[0].plot(x_line, p[0])
    axarr[0].set_ylabel('CO2 Emissions')
    axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
    axarr[1].plot(x_line, p[1])
    axarr[1].set_xlabel('Year')
    axarr[1].set_ylabel('Relative temperature')

