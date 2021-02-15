# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:56:59 2019

"""


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select 
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

import time
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
from dateutil import relativedelta
import csv
import xlsxwriter
import os
import glob
import zipfile

import warnings
os.chdir(working_path)# change working directory


##test chrome driver ch
#browser = webdriver.Chrome("C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe")
#browser.get("http://www.baidu.com/")
def check_stationary(ts):
    """
    This function checks if a time series is stationary in a naive way:
    If the value at anypoint is larger than 250, return false, else return true.
    """
    for i in range(len(ts)):
        for j in range(len(ts[0])):
            if(abs(ts[i][j])>250):
                return 0;
    return 1;     
def check_stationary_for_diff(ts):
    """
    This function checks if a time series of first order differences is stationary in a naive way:
    If the value at anypoint is larger than 15, return false, else return true.
    """
    for i in range(len(ts)):
        for j in range(len(ts[0])):
            if(abs(ts[i][j])>15):
                return 0;
    return 1;     
def take_diff(ts):
    """
    This function takes a difference for a time series. 
    """
    l = []
    for i in range(len(ts)-1):
        l.append(ts[i+1]-ts[i])
    return l

def take_diff_inv(ts):
    """
    This function do the opposite of taking a difference, it converts a difference time series to a level time series.
    """
    l = [np.array([0 for _ in range(len(ts[0]))])]
    for i in range(len(ts)):
        l.append(l[-1]+ts[i])
    return l

def take_diff_inv_for_seasonal(ts,s):
    """
    This function takes difference for seasonal model with s as the season, i.e s = 12 for yearly seasonality , s = 4 for quarterly, etc
    """
    l = [np.array([0 for _ in range(len(ts[0]))]) for _ in range(s)]
    for i in range(len(ts)):
        l.append(l[-s]+ts[i])
    return l;

class ARIMA:
    """
    This class defines an ARIMA model, with default parameters (p,d,q) = (2,0,1), n=3 is the default number of variables, 
    n_loops = 500 is the default simulation periods.
    """
    
    def __init__(self,p=2,d=0,q=1,n=3,n_loops=500):
        self.p = p;
        self.d = d;
        self.q = q;
        self.n = n;
        self.n_loops = n_loops;
    def run(self):
        """
        This function simulates the ARIMA(p,d,q) process and store results in Y.
        
        """
        Y = []
        W = []
        # W is white noise 
        for _ in range(self.p):
            #generate a random initial Y value
            Y.append(np.random.normal(0,2,self.n))
        for _ in range(self.q):
            #generate a random initial while noise
            W.append(np.random.normal(0,0.5,self.n))
        L_AR,L_MA = self.genCoefs()

        for i in range(self.n_loops):
            ep = np.random.normal(0,0.5,self.n)

            W.append(ep)
            y = ep

            for i_AR in range(self.p):
                y = y + L_AR[i_AR].dot(Y[-(i_AR+1)])

            for i_MA in range(self.q):
                y = y + L_MA[i_MA].dot(W[-(i_MA+1)])



            Y.append(y)

        for i in range(self.d):
            Y = take_diff_inv(Y)
        if(check_stationary(Y)==0):
            Y = self.run()
        return Y
    
        
    def genCoefs(self,MAX = 0.6):
        """
        # For the AR terms, the coeff of higher order lags is always smaller than lower order ones.
        # s denotes seasonal PDQ
        """
        n = self.n
        X_MAX = [MAX for _ in range(n)]

        L_AR = [self.genCoefs_AR(X_MAX)]
        

        for i in range(self.p-1):
            L_AR.append(self.genCoefs_AR(np.diag(L_AR[i]/2)))

        L_MA = []
        for i in range(self.q):
            L_MA.append(self.genCoefs_MA())
        return L_AR,L_MA
    def genCoefs_AR(self,X_MAX):
        n = self.n
        """
         This function generates a nxn matrix (n is # of input variables, i.e. GDP, CPI, VIX, etc.) 
         X_MAX is the maximum possible value for self correlation coefficent to be. X_MAX is a nx1 array
         The diagonal is the AR lag of the variable itself,i.e.X[3][3] is y3_t = X[3][3] * y3_t-1,X[3][2] is y3_t = X[3][2]*y2_t-1
         Hence, the elements on the diagonal is assumed to be larger than all the values in its row and col. 
         The interpretation is the influence on itself must be larger than the influence it gives to others or others give to it.
        """
        X = np.diagflat([[np.random.uniform(X_MAX[i]/2,X_MAX[i])] for i in range(n)])
        v = np.diag(X)
        for i in range(1,n):

            l = []
            for j in range(len(v)-i):
                l.append(min(abs(v[j]),abs(v[j+i])))
            X = X + np.diagflat([random.choice([np.random.uniform(-l[k], -0.05), np.random.uniform(0.05,l[k])]) for k in range(n-i)],i)
            X = X + np.diagflat([random.choice([np.random.uniform(-l[k], -0.05), np.random.uniform(0.05,l[k])]) for k in range(n-i)],-i)
        return X
    
    def genCoefs_MA(self):
        n = self.n
        X = np.array([[random.choice([np.random.uniform(0.2,0.6),np.random.uniform(-0.6,-0.2)]) for _ in range(n)] for _ in range(n)])
        return X.transpose()
    
class SARIMA(ARIMA):
    """
    Seasonal ARIMA extends ARIMA class
    """
   
        
    def __init__(self,p=2,d=0,q=1,s=12,P=1,D=0,Q=1,n=3,n_loops=500):
        super().__init__(p,d,q,n,n_loops);
        self.s = s;
        self.P = P;
        self.D = D;
        self.Q = Q;
        
    def run_with_Seasonality(self):
        Y = []
        W = []
        for _ in range(self.p+self.s+self.P):
            Y.append(np.random.normal(0,2,self.n))
        for _ in range(self.q+self.s+self.Q):
            W.append(np.random.normal(0,0.5,self.n))
        L_AR,L_MA = self.genCoefs();
        L_AR_s,L_MA_s = self.genCoefs(MAX=0.2);
        for i in range(self.n_loops):
            ep = np.random.normal(0,0.5,self.n)
            W.append(ep)
            y = ep

            for i_AR in range(self.p):
                y = y + L_AR[i_AR].dot(Y[-(i_AR+1)])
            for i_sAR in range(self.P):
                y = y + L_AR_s[i_sAR].dot(Y[-(i_sAR+1+self.s)])

            for i_MA in range(self.q):
                y = y + L_MA[i_MA].dot(W[-(i_MA+1)])
            for i_sMA in range(self.Q):
                y = y + L_MA_s[i_sMA].dot(W[-(i_sMA+1+self.s)])
            Y.append(y)

        for i in range(self.d):
            Y = take_diff_inv(Y)
        for i in range(self.D):
            Y = take_diff_inv_for_seasonal(Y,s)
        if(check_stationary(Y)==0):
            Y = self.run_with_Seasonality();
        
        return Y
    

from datetime import date

def seperate_generated(ts, freq):
#    date = init_date
    today = date.today()
    cripath = r'\\dirac\CRI3\OfficialTest_AggDTD_SBChinaNA\ProductionData\Historical'
    files = [i for i in os.listdir(cripath) if len(i)==6 ]
    files.sort()
    cridate = int(files[-1])
    if min(freq) == 1:
        for i in range(len(ts)):
            if cridate < today.year*100+today.month:
                if(ts[i][0] == np.floor(cridate/100) and ts[i][1]==cridate-np.floor(cridate/100)*100):
                    return ts[:(i+1)],ts[(i+1):]
            else:
                if(ts[i][0] == today.year and ts[i][1]==today.month):
                    return ts[:i],ts[i:]
    elif min(freq) == 0:
        idx = np.where(today.year*100+today.month==(ts[:,0]*100+ts[:,1]))[0][0]
        last_idx = np.where(ts[:idx,2+freq.index(0)] != np.nan)[0][-1]
        Year = ts[last_idx+1,0]
        Month = ts[last_idx+1,1]
        for i in range(len(ts)):
            if cridate < Year*100+Month:
                if(ts[i][0] == np.floor(cridate/100) and ts[i][1]==cridate-np.floor(cridate/100)*100):
                    return ts[:(i+1)],ts[(i+1):]
            else:
                if(ts[i][0] == Year and ts[i][1]== Month):
                    return ts[:i],ts[i:]
    else:
        for i in range(len(ts)):
            if(ts[i][0] == today.year and ts[i][1]== 1):
                return ts[:i],ts[i:]       
        
        
def plot_var_x(x,Y):
    l = []
    for i in range(len(Y)):
        l.append(Y[i][x])
    plt.plot(l)
    
def add_time(Y,freq,n,init_date = datetime(1993,1,3)):
#    date = init_date
#    X = np.zeros((len(Y),n)).tolist();
#    for i in range(len(X)):
#        X[i] = np.insert(X[i],0,date.month)
#        X[i] = np.insert(X[i],0,date.year)
#        date = date+relativedelta.relativedelta(months=1)        
    
    ts = pd.date_range(init_date,periods=len(Y),freq='M')
    X = np.zeros((len(Y),n+2))
    X[X==0] = np.nan
    X[:,0] = ts.year
    X[:,1] = ts.month
    Y = np.array(Y)
    for j in range(len(freq)):
        if freq[j] == 0: # quarterly
            X[2::3,j+2] = Y[2::3,j]
        elif freq[j] == -1: # yearly
            X[11::12,j+2] = Y[11::12,j]
        else: # monthly
            X[:,j+2] = Y[:,j]
    X = np.where(np.isnan(X),None,X)
    return X

def output(mod1,Y,freqList, mode = 'diff',path = r"C:\Users\\"):
    n = mod1.n
    headers = [["example"],[''],["This Frequency provides the information whether the training macro-economic scenarios used are reported on a monthly basis or a quarterly basis or a yearly basis."],['The value "1" means "Monthly"; "0" means "Quarterly"; and the value "-1" means "Yearly".'],["If it is on a quarterly basis;the data should be reported in Month 3 6 9 12 while blank need be reported in other months."], ["If it is on a yearly basis; the data should be reported in Month 12 while blank need be reported in Month 1-11"], ["Growth rate on a monthly basis should be MoM growth rate (non-annualized); on a quarterly basis should be QoQ growth rate (non-annualized); on a yearly basis should be YoY growth rate."]]
#    frequency = ['Frequency','']+[1 for i in range(n)]
    if freqList==0:
        freq = np.random.randint(-1,2,n).tolist()
        macro = ['Macro Type',''] + np.random.randint(-1,2,n).tolist()
    else:
        freq = freqList[0]
        macro = ['Macro Type',''] + freqList[1]
    frequency = ['Frequency','']+freq
    headers2 = [["This Macro Type provides the information that for each country whether the training macroeconomic scenario is the change (growth rate/difference) or the level."],['The value "1" means "Change (Growth Rate %)";  the value "0" means "Change (Difference)"; the value "-1" means "Level".']]
#    if(mode=='diff'):
#        macro = ['Macro Type',''] + [0 for i in range(n)]
#    elif(mode=='level'):
#        macro = ['Macro Type',''] + [-1 for i in range(n)]    
    data_header = ['year','month'] + [str('V'+str(i+1)) for i in range(n)]
    TS = add_time(Y,freq,n)
    Y_prev, Y_after = seperate_generated(TS,freq)
    ## this part is for official website, only accept csv file    
    with open(path+"\\first_csv.csv",'w+',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(headers)
        writer.writerow(frequency)
        writer.writerow([])
        writer.writerows(headers2)
        writer.writerow(macro)
        writer.writerow([])
        writer.writerow(data_header)
        writer.writerows(Y_prev)
    csvFile.close()
    ## this part is for test website, only accept xlsx file
    first_file = path+"\\first_xlsx.xlsx"
    # data = pd.DataFrame(Y_prev)
    Y_prev = pd.DataFrame(Y_prev)
    writer = pd.ExcelWriter(first_file, engine='xlsxwriter')
    Y_prev.to_excel(writer, sheet_name='first', startrow=14, header=False,index = False)
    worksheet = writer.sheets['first']
    for i in range(len(headers)):
        worksheet.write(i, 0, headers[i][0])        
    for col_num in range(len(frequency)):
        worksheet.write(7, col_num, frequency[col_num])
    worksheet.write(8, 0, '')
    for i in range(len(headers2)):
        worksheet.write(9+i, 0, headers2[i][0])
    for col_num in range(len(macro)):
        worksheet.write(11, col_num, macro[col_num])
    worksheet.write(12, 0, '')
    for col_num in range(len(data_header)):
        worksheet.write(13, col_num, data_header[col_num])
    writer.close()
#    now = datetime.now()
    #senario_file = path+"save_as_xlsx_template_{}-{}-{}T{}-{}-{}.xlsx".format(now.year,now.month,now.day,now.hour,now.minute,now.second)
、    senario_file = path+"\\save_as_xlsx_template.xlsx"
    senario_headers = [["(i) Please specify the frequency of each selected stress variable, where '1' for monthly data(MoM), '0' for quarter-end data(QoQ), '-1' for year-end data(YoY)"],["(ii) Please fill the time series of the selected stress variables. The quarter-end data will be filled in months 3,6,9,12, and month 12 for the year-end data"],["(iii) Please refer to Table 6 in BuDA White Paper for information on description of the Provided Macroeconomic Variables"],[]]
    frequency = [['','']+data_header[-n:], ['','frequency']+freq,[]]

#    with open(senario_file,'w+',newline='') as csvFile:
#        writer = csv.writer(csvFile)
#        writer.writerows(senario_headers)
#        writer.writerows(frequency)
#        writer.writerow(data_header)
#        writer.writerows(Y_after)
#    csvFile.close()
    data = pd.DataFrame(Y_after)
    writer = pd.ExcelWriter(senario_file,
                        engine='xlsxwriter')
    data.to_excel(writer, sheet_name='Scenario 1', startrow=8, header=False,index = False)
    worksheet = writer.sheets['Scenario 1']
    # Write the column headers with the defined format.
    for i in range(len(senario_headers)-1):
        worksheet.write(i, 0, senario_headers[i][0])
        
    for row_num in range(len(frequency)):
        for col_num in range(len(frequency[row_num])):
            worksheet.write(row_num+4, col_num, frequency[row_num][col_num])
        
    for col_num in range(len(data_header)):
        worksheet.write(7, col_num, data_header[col_num])
    
    writer.close()
    return senario_file

    
def read_config(level):
    """ read config file containing params """
    
    file_pth = working_path+'buda_config.xlsx'
    sheets = pd.ExcelFile(file_pth).sheet_names
    config = {}
    for i in sheets:
        df = pd.read_excel(file_pth,i,index_col=0)
        df.columns = [x.lower() for x in df.columns]
        df = df[df.level==level] #change the variable level you want
        df.astype(str)
        config[i]  = df
    return config

def random_economy(continent_id):
    file_pth = working_path+'buda_config.xlsx'
    df = pd.read_excel(file_pth,"economy",index_col=0)
    economies_list = list(df[df.parent_id == int(continent_id)].economy_id) 
    from numpy import random
    rand_list = random.randint(int(len(economies_list)), size = max(1,int(0.5*len(economies_list))))
    economies = list(set([str(economies_list[x]) for x in rand_list]))## delete duplicate number
    return economies
    
def random_combination(config):
    """ randomly assign params"""
    
    lst = [config[x][f'{x}_id'] for x in config]
    from itertools import product
    temp = list(product(lst[0],lst[1],lst[0],lst[2],lst[3]))
    from numpy import random
    rand_num = random.randint(int(len(temp)),size=int(0.01*len(temp)))
    combination = [tuple(map(str,temp[x])) for x in rand_num]##turn to str
    return combination


def log_in(version,account,passwd):
    """ log in Buda website """
    options = webdriver.ChromeOptions()
    options.add_argument("--auto-open-devtools-for-tabs")
    options.binary_location = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    if version == 81:
        driver = webdriver.Chrome(working_path+'chromedriver81.exe', chrome_options=options)
    else: # version = 83
        driver = webdriver.Chrome(working_path+'chromedriver83.exe', chrome_options=options)
    driver.get('https://client.rmicri.org/budatoolkit') # the official website
    driver.find_element_by_name('email').send_keys(f'{account}')
    driver.find_element_by_name('password').send_keys(f'{passwd}')
    driver.find_element_by_tag_name('button').click() 
    return driver
  
    
def User_specified(driver,itest, filename):
    """ add params """    
    ## step1: testing_portfolio
    Select(driver.find_element_by_id('continent')).select_by_value(itest[0])
#   Select(driver.find_element_by_id('economy')).select_by_value(combination[1])
    economy1 = random_economy(itest[0])
    for i in range(0,len(economy1)):
           Select(driver.find_element_by_id('economy')).select_by_value(economy1[i])
           driver.find_element_by_id('btn-addecontoken').click() 
           time.sleep(1)
    
    Select(driver.find_element_by_name('industry')).select_by_value(itest[1])
    driver.find_element_by_id('btn-addindustoken').click() 
    driver.find_element_by_css_selector(".btn.next-step.action-button").click()
    time.sleep(3)
    
    ## step2: testing_scenario
# =============================================================================
#     Select(driver.find_element_by_id('macro-continent')).select_by_value(itest[2])
# #    Select(driver.find_element_by_id('macro-economy')).select_by_value(combination[2])
#     Select(driver.find_element_by_id('macro-scenarios')).select_by_value(itest[3])
#     driver.find_element_by_id('btn-addmacrotoken').click()
#     Select(driver.find_element_by_id('constant-scenarios')).select_by_value(itest[4]) ##other stress variable of interest
#     driver.find_element_by_id('btn-addconstmacrotoken').click()
#     time.sleep(3)
# =============================================================================
    # add User Supplied Stress-testing Variables
    driver.find_element_by_id('user-macros').click()
    file_path_1 =  r'\first_csv.csv'
    driver.find_element_by_id("user-macro-modal-trainfile").send_keys(file_path_1)
    driver.find_element_by_id("user-macro-modal-trainfile").submit()
    time.sleep(10)
    driver.find_element_by_id('check-sample-btn').click() # comfirm button
    time.sleep(40)
    
    ## step3: test
    driver.find_element_by_id('generate-file-btn').click()   # download file button
    time.sleep(30)
    driver.find_element_by_id('btn-macrotest').click()

    
    file_path_2 =  filename
    driver.find_element_by_id("run-modal-infile").send_keys(file_path_2)
    driver.find_element_by_id("run-modal-infile").submit()
    ## upload check not yet
    time.sleep(5)
    driver.find_element_by_css_selector(".btn.next.action-button").click()
    time.sleep(3)
    #random_date = datetime(2018,10,1).strftime("%Y-%m") # can be set randomly
   # driver.find_element_by_id('datepicker').send_keys(f'{random_date}')
    Select(driver.find_element_by_id('sample-period-choices')).select_by_value("sample-period-upto")
    horizon_month = 12
    driver.find_element_by_id('horizon-input').clear()
    driver.find_element_by_id('horizon-input').send_keys(f'{horizon_month}')
    driver.find_element_by_id('submit-form-btn').click()
    
    
    ## step4: download final result
    #driver.find_element_by_css_selector('.fas.fa-download').click()

    ## advance setting not yet

def backtest(driver,itest):
     """ add params """    
    ## step1: testing_portfolio
     Select(driver.find_element_by_id('continent')).select_by_value(itest[0])
     economy1 = random_economy(itest[0])
     for i in range(0,len(economy1)):
           Select(driver.find_element_by_id('economy')).select_by_value(economy1[i])
           driver.find_element_by_id('btn-addecontoken').click() 
           time.sleep(1)
    
     Select(driver.find_element_by_name('industry')).select_by_value(itest[1])
     driver.find_element_by_id('btn-addindustoken').click() 
     driver.find_element_by_css_selector(".btn.next-step.action-button").click()
     time.sleep(3)
          
    ## step2: testing_scenario
     Select(driver.find_element_by_id('macro-continent')).select_by_value(itest[2])
     economy2 = random_economy(itest[2])
     for i in range(0,len(economy2)):
           Select(driver.find_element_by_id('macro-economy')).select_by_value(economy2[i])
           Select(driver.find_element_by_id('macro-scenarios')).select_by_value(itest[3])
           driver.find_element_by_id('btn-addmacrotoken').click()
           time.sleep(1)
#    Select(driver.find_element_by_id('macro-economy')).select_by_value(combination[2])
     #Select(driver.find_element_by_id('macro-scenarios')).select_by_value(itest[3])
     #driver.find_element_by_id('btn-addmacrotoken').click()
     Select(driver.find_element_by_id('constant-scenarios')).select_by_value(itest[4]) ##other stress variable of interest
     driver.find_element_by_id('btn-addconstmacrotoken').click()
     time.sleep(5)
     driver.find_element_by_id('check-sample-btn').click() # comfirm button
     time.sleep(12)
     
    ## step3: backtest
#     driver.find_element_by_id('generate-file-btn').click()   # download file button
#     time.sleep(10)
    ## upload check not yet
     driver.find_element_by_id('actual-realize').click() 
     driver.find_element_by_css_selector(".btn.next.action-button").click()
     time.sleep(3)
    #random_date = datetime(2018,10,1).strftime("%Y-%m") # can be set randomly
    #driver.find_element_by_id('datepicker').send_keys(f'{random_date}')
     Select(driver.find_element_by_id('sample-period-choices')).select_by_value("sample-period-upto")
     horizon_month = 12
     driver.find_element_by_id('horizon-input').clear()
     driver.find_element_by_id('horizon-input').send_keys(f'{horizon_month}')
     driver.find_element_by_id('submit-form-btn').click()
#     now = datetime.now()
#     s_file = "save_as_xlsx_template_{}-{}-{}T{}-{}-{}.xlsx".format(now.year,now.month,now.day,now.hour,now.minute,now.second) ##different from the time shown on the file 
#     return s_file
     #driver.find_element_by_css_selector('.fas.fa-download').click()
      
    ## advance setting not yet

def test(mode,itest,combination,driver,filename):
     #user-specified scenario test
    if mode == "user_specified":
        try:
            User_specified(driver, itest, filename)
        except:
             rand_num = random.randint(0,len(combination)-1)
             itest =  combination[rand_num]
             driver.get('https://client.rmicri.org/budatoolkit')
             test(mode,itest,combination,driver,filename)         
    #backtest
    elif mode == "backtest" : 
        try:
            backtest(driver,itest)  
        except:
            rand_num = random.randint(0,len(combination)-1)
            itest =  combination[rand_num]
            driver.get('https://client.rmicri.org/budatoolkit')
            test(mode,itest,combination,driver,filename)
    else:
        print("please choose one standard mode!!")
            
def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    for names in zip_file.namelist():
        zip_file.extract(names,file_name + "_files/")
    zip_file.close()
    
def result_examination(download_path, driver):
    #driver.implicitly_wait(1200)
    ##step1: download final result
    #driver.find_element_by_css_selector('.fas.fa-download').click()
    element = driver.find_element_by_css_selector('.fas.fa-download')
    driver.execute_script("arguments[0].click();", element)

    #download_path = r'C:\Users\Downloads'  ## you may change it  to your own path
    time.sleep(7)
    ##step2: extract zip file
    #now = datetime.now() 
    d = date.today()
    folder = download_path + "\Test_*{}{}{}*.zip".format(d.strftime('%d'),d.strftime('%m'),d.strftime('%Y')) ##the test result generated today
    files = glob.glob(folder)
    names = [0]*len(files)                                                                                                        
    for ifile in range(0,(len(files)-1)):
        names[ifile] = int(os.path.split(files[ifile])[1][-10:-4])
    zip_path = files[names.index(max(names))]
    un_zip(zip_path) 
    
    ##step3: examine file size
    
    file_path = zip_path+'_files'
    os.chdir(file_path)
    file_list = os.listdir(file_path)
    if len(file_list)<10:
        warnings.warn('some files are lost!!!',RuntimeWarning)
        
    os.listdir(file_path)
    for i in range(0,len(file_list)):
        if os.path.getsize(os.listdir(file_path)[i])==0:
            warnings.warn('some files are empty!!!',RuntimeWarning)
    
    
def Buda_autotest(mode,version,level,freqList=0):
# =============================================================================
#     make sure you have chrome C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
#     enter 'about::version' to check your version: 
#     version = 81 or version = 83
#     mode = "user_specified" or "backtest"
#     level = 1 or 2 or 3 (so far level cannot equal to 3)
#     freqList = [[1,0,1,1,0],[-1,0,0,-1,1]] (a big list containing two lists of length 4 to control frequency)
# =============================================================================
    
    ##generate test set
    config = read_config(level) 
    combination = random_combination(config)
    rand_num = random.randint(0,len(combination)-1)
    itest=  combination[rand_num]
    driver = log_in(version)
    time.sleep(2)
    driver.find_element_by_link_text("link").click()
    time.sleep(8)

   
    ####generate stress test variables
    path = r'\Stresstest_Variable'
    #mod1 = ARIMA()
    #mod1 = SARIMA()
    #mod1 = ARIMA(2,1,2,5,500)
    mod1 = SARIMA(2,1,1,12,1,0,1,5,500)
    if(type(mod1)==SARIMA):
        Y = mod1.run_with_Seasonality()
    elif(type(mod1)==ARIMA):
        Y = mod1.run()
    plt.plot(Y)
    plt.show()
    Y_diff = take_diff(Y)
    plt.plot(Y_diff)
    filename = output(mod1,Y_diff,freqList,path = path)
    
 
    ##test
    test(mode,itest,combination,driver,filename)
    
    #examine the result
    download_path = r'C:\Users\Downloads' ## you may change it  to your own path
    result_examination(download_path, driver)

 # #     for itest in combination:
 # #         print(f'{datetime.now()}: testing {count}th starts...')
 # #         test(driver,itest)
 # #         print(f'{datetime.now()}: testing {count}th finishes.../n')
