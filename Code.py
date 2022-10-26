from multiprocessing import pool
from matplotlib import image
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import streamlit.components.v1 as stc
import matplotlib.pyplot as plt
import seaborn as sns
import setuptools
import os
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.stats.weightstats as statsmod
from statsmodels.formula.api import ols #For  n-way ANOVA
from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd #For performing the Tukey-HSD test
from statsmodels.stats.multicomp import MultiComparison #To compare the levels of independent variables
population_mean = '\u03BC'
population_standard_deviation = '\u03C3'
population_proportion = '\u03C0'
sample_proportion = 'p\u0302'
from PIL import Image
Confidence_Interval = Image.open('Normal Distribution.png')
Hypothesis_testing = Image.open('new.png')
ANOVA = Image.open('One-way anova.png')

html_temp = """
        <div style="background-color:#9FE2BF;padding:10px;border-radius:10px">
		<h1 style="color:black;text-align:center;">Inferential Statistics</h1>
		</div>"""
menu = ["Home","Confidence Interval(CI) Estimation","Hypothesis testing"]

choice = st.sidebar.selectbox("Menu",menu,index=0)
stc.html(html_temp)

def ci_mu_sigma_known(pop_sd,sample_mean,sample_size,ci_level):
    if pop_sd>0 and sample_mean>0 and sample_size>30 and ci_level>0:
        alpha = 1 - ci_level
        Z_alpha_by_2 = np.abs(stats.norm.ppf(alpha/2,0,1))
        term1 = pop_sd/sample_size**0.5
        lower_limit = sample_mean - (Z_alpha_by_2*term1)
        upper_limit = sample_mean + (Z_alpha_by_2*term1)
        return lower_limit,upper_limit
    elif pop_sd>0 and sample_mean>0 and sample_size>0 and sample_size<30 and ci_level>0:
        alpha = 1 - ci_level
        t_alpha_by_2 = np.abs(stats.t.ppf(alpha/2,sample_size-1))
        term1 = pop_sd/sample_size**0.5
        lower_limit = sample_mean - (t_alpha_by_2*term1)
        upper_limit = sample_mean + (t_alpha_by_2*term1)
        return lower_limit,upper_limit
        pass

def ci_mu_sigma_unknown(sample_sd,sample_mean,sample_size,ci_level):
    if sample_sd>0 and sample_mean>0 and sample_size>0 and ci_level>0:
        alpha = 1 - ci_level
        t_alpha_by_2 = np.abs(stats.t.ppf(alpha/2,sample_size-1))
        term1 = sample_sd/sample_size**0.5
        lower_limit = sample_mean - (t_alpha_by_2*term1)
        upper_limit = sample_mean + (t_alpha_by_2*term1)
        return lower_limit,upper_limit

def ci_prop_pop_sample(X,sample_size,ci_level):
    alpha = 1 - ci_level
    z_alpha_by_2 = np.abs(stats.norm.ppf(alpha/2,0,1))
    p = X/sample_size
    term1 = (p*(1-p))
    term2 = (term1/(sample_size))**0.5
    lower_limit = p - (z_alpha_by_2*term2)
    upper_limit = p + (z_alpha_by_2*term2)
    return lower_limit,upper_limit

def ci_prop_pop_sample_p_given(p_given,ci_level):
    alpha = 1 - ci_level
    z_alpha_by_2 = np.abs(stats.norm.ppf(alpha/2,0,1))
    p = p_given
    term1 = (p*(1-p))
    term2 = (term1/(sample_size))**0.5
    lower_limit = p - (z_alpha_by_2*term2)
    upper_limit = p + (z_alpha_by_2*term2)
    return lower_limit,upper_limit

def confidence_interval_mean_difference(X1bar,X2bar,S1,S2,n1,n2,ci_level):
    alpha = 1 - ci_level
    Sp2 = (((n1-1)*(S1**2)+(n2-1)*(S2**2))/((n1-1)+(n2-1)))
    t_alpha_by_2 = np.abs(stats.t.ppf(alpha/2,n1+n2-2))
    term1 = ((1/n1) + (1/n2))
    lower_limit = (X1bar - X2bar) - (t_alpha_by_2)*((Sp2*term1)**0.5)
    upper_limit = (X1bar - X2bar) + (t_alpha_by_2)*((Sp2*term1)**0.5)
    return lower_limit,upper_limit

def z_test(Xbar,Mu,sigma,n,alpha,test_type):
    z_stat = (Xbar - Mu)/((sigma/(n)**0.5))
    if test_type=="Left tail test":
        critical_region_value = stats.norm.ppf(alpha,0,1)
        p_value = stats.norm.cdf(z_stat,0,1)
        if z_stat<critical_region_value:
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            return z_stat,p_value,'Do not reject the null hypothesis'
    if test_type=="Right tail test":
        critical_region_value = stats.norm.ppf(alpha,0,1)
        p_value = 1-stats.norm.cdf(z_stat,0,1)
        if z_stat>critical_region_value:
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            return z_stat,p_value,'Do not reject the null hypothesis'
    if test_type=="Two tail test":
        critical_region_value = stats.norm.ppf((alpha/2),0,1)
        if z_stat<critical_region_value:
            p_value = stats.norm.cdf(z_stat,0,1)
            return z_stat,p_value*2,'Reject the null hypothesis'
        elif(z_stat>np.abs(critical_region_value)):
            p_value = 1-stats.norm.cdf(z_stat,0,1)
            return z_stat,p_value*2,'Reject the null hypothesis'
        else:
            p_value2 = stats.norm.cdf(z_stat,0,1)
            p_value1=1-stats.norm.cdf(z_stat,n-1)
            p_value=min(p_value1,p_value2)
            return z_stat,p_value*2,'Fail to reject the null hypothesis'

def t_test(Xbar,Mu,S,n,alpha,test_type):
    t_stat = (Xbar - Mu)/((S/(n)**0.5))
    if test_type=="Left tail test":
        critical_region_value = stats.t.ppf(alpha,n-1)
        p_value = stats.t.cdf(t_stat,n-1)
        if t_stat<critical_region_value:
            return t_stat,p_value,'Reject the null hypothesis'
        else:
            return t_stat,p_value,'Do not reject the null hypothesis'
    if test_type=="Right tail test":
        critical_region_value = np.abs(stats.t.ppf(alpha,n-1))
        p_value = 1-stats.t.cdf(t_stat,n-1)
        if t_stat>critical_region_value:
            return t_stat,p_value,'Reject the null hypothesis'
        else:
            return t_stat,p_value,'Do not reject the null hypothesis'
    if test_type=="Two tail test":
        critical_region_value = stats.t.ppf((alpha/2),n-1)
        if t_stat<critical_region_value:
            p_value = stats.t.cdf(t_stat,n-1)
            return t_stat,p_value*2,'Reject the null hypothesis'
        elif(t_stat>np.abs(critical_region_value)):
            p_value = 1-stats.t.cdf(t_stat,n-1)
            return t_stat,p_value*2,'Reject the null hypothesis'
        else:
            p_value2 = stats.t.cdf(t_stat,n-1)
            p_value1=1-stats.t.cdf(t_stat,n-1)
            p_value=min(p_value1,p_value2)
            return t_stat,p_value*2,'Fail to reject the null hypothesis'
        
def one_sample_proportion_ztest(sample_prop,p,n,alpha,test_type):
    nr = sample_prop - p
    term1 = ((p*(1-p))/(n))**0.5
    z_stat = nr/term1
    if test_type == 'Left tail':
        critical_region_value = stats.norm.ppf(alpha,0,1)
        p_value = stats.norm.cdf(z_stat,0,1)
        if z_stat<critical_region_value:
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            return z_stat,p_value,'Do not reject the null hypothesis'
    elif test_type=="Right tail test":
        critical_region_value = np.abs(stats.norm.ppf(alpha,0,1))
        p_value = 1-stats.norm.cdf(z_stat,0,1)
        if z_stat>critical_region_value:
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            return z_stat,p_value,'Do not reject the null hypothesis'
    else:   
        critical_region_value = stats.norm.ppf((alpha/2),0,1)
        if z_stat<critical_region_value:
            p_value = stats.norm.cdf(z_stat,0,1)
            return z_stat,p_value,'Reject the null hypothesis'
        elif(z_stat>np.abs(critical_region_value)):
            p_value = 1-stats.norm.cdf(z_stat,0,1)
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            p_value2 = stats.norm.cdf(z_stat,0,1)
            p_value1=1-stats.norm.cdf(z_stat,n-1)
            p_value=min(p_value1,p_value2)
            return z_stat,p_value,'Fail to reject the null hypothesis'

def pooled_variance_t_test(Xbar1,Xbar2,S1,S2,n1,n2,alpha):
    hypothesized_mean_difference = int(input("Enter the hypothesized difference in means for the two populations"))
    Xbar1 = float(input("Enter the mean for sample 1 :"))
    Xbar2 = float(input("Enter the mean for sample 2 :"))
    S1 = float(input("Enter the standard deviation for sample 1 :"))
    S2 = float(input("Enter the standard deviation for sample 2 :"))
    n1 = int(input("What is the size for sample 1?"))
    n2 = int(input("What is the size for sample 2?"))
    alpha = float(input("What is the level of significance?"))
    pooled_variance = (((n1-1)*(S1**2)+(n2-1)*(S2**2))/((n1-1)+(n2-1)))**0.5
    Nr = ((Xbar1 - Xbar2) - (hypothesized_mean_difference))
    Dr = pooled_variance*(((1/n1)+(1/n2))**0.5)
    t_stat = Nr/Dr
    test_type = input("Which test do you want to conduct A) Left tail test B) Right tail test C) Two tail test. [A/B/C]?:")
    if test_type=="A":
        critical_region_value = stats.t.ppf(alpha,n1+n2-1)
        p_value = stats.t.cdf(t_stat,n1+n2-1)
        if t_stat<critical_region_value:
            return t_stat,p_value,'Reject the null hypothesis'
        else:
            return t_stat,p_value,'Fail to reject the null hypothesis'
    if test_type=="B":
        critical_region_value = np.abs(stats.t.ppf(alpha,n1+n2-1))
        p_value = 1-stats.t.cdf(t_stat,n1+n2-1)
        if t_stat>critical_region_value:
            return t_stat,p_value,'Reject the null hypothesis'
        else:
            return t_stat,p_value,'Reject the null hypothesis'
    if test_type=="C":
        critical_region_value = stats.t.ppf((alpha/2),n1+n2-1)
        if t_stat<critical_region_value:
            p_value = stats.t.cdf(t_stat,n1+n2-1)
            return t_stat,p_value,'Reject the null hypothesis'
        elif(t_stat>np.abs(critical_region_value)):
            p_value = 1-stats.t.cdf(t_stat,n1+n2-1)
            return t_stat,p_value,'Reject the null hypothesis'
        else:
            p_value2 = stats.t.cdf(t_stat,n1+n2-1)
            p_value1=1-stats.t.cdf(t_stat,n1+n2-1)
            p_value=min(p_value1,p_value2)
            return z_stat,p_value,'Fail to reject the null hypothesis'

def difference_in_proportion(X1,X2,n1,n2,hypothesized_mean_difference):
    p1 = X1/n1
    p2 = X2/n2
    pooled_prop = ((X1+X2)/(n1+n2))
    nr = (p1-p2) - (hypothesized_mean_difference)
    dr = ((pooled_prop)*(1-pooled_prop)*((1/n1)+(1/n2)))**0.5
    z_stat = nr/dr
    if test_type == 'Left tail test':
        critical_region_value = stats.norm.ppf(alpha,0,1)
        p_value = stats.norm.cdf(z_stat,0,1)
        if z_stat<critical_region_value:
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            return z_stat,p_value,'Do not reject the null hypothesis'
    elif test_type=="Right tail test":
        critical_region_value = np.abs(stats.norm.ppf(alpha,0,1))
        p_value = 1-stats.norm.cdf(z_stat,0,1)
        if z_stat>critical_region_value:
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            return z_stat,p_value,'Do not reject the null hypothesis'
    else:   
        critical_region_value = stats.norm.ppf((alpha/2),0,1)
        if z_stat<critical_region_value:
            p_value = stats.norm.cdf(z_stat,0,1)
            return z_stat,p_value,'Reject the null hypothesis'
        elif(z_stat>np.abs(critical_region_value)):
            p_value = 1-stats.norm.cdf(z_stat,0,1)
            return z_stat,p_value,'Reject the null hypothesis'
        else:
            p_value2 = stats.norm.cdf(z_stat,0,1)
            p_value1=1-stats.norm.cdf(z_stat,0,1)
            p_value=min(p_value1,p_value2)
            return z_stat,p_value,'Fail to reject the null hypothesis'
    

def validate_file(file):
    filename = file.name
    name,ext = os.path.splitext(filename)
    if ext in ('.csv','.xlsx'):
        return ext
    else:
        return False


if choice == 'Home':
    col1,col2 = st.columns(2)
    with col1:
        st.image(Confidence_Interval)
    with col2:
        st.image(Hypothesis_testing)
    st.markdown(""" ## About the app
    This application is designed for doing Inferential Statistics. 
    One can use this to estimate the confidence interval as well as to do hypothesis testing. 
    This application uses some of the parametric tests to perform hypothesis testing. 
    Assumptions made for those tests are also mentioned. 
    The input data can be in the form of one number or an array.

    ##### The application has two parts:
    - Confidence Interval Estimation
    - Hypothesis testing

    Scroll down below for more details about the app

    """)

    st.subheader("Confidence Interval Estimation")

    with st.expander('Expand to learn about the coverage of confidence interval estimation'):
        st.markdown(""" 
        #### Confidence interval estimation
        This part deals with answering the question: 
        **With what confidence can you say that your parameter of interest lies in a certain interval?**

        With the app you will be able to:
        1.	Estimate the confidence interval for population mean(\u03BC) when population standard deviation(\u03C3) is known
        2.	Estimate the confidence interval for population mean(\u03BC) when population standard deviation(\u03C3) is unknown
        3.	Estimate the confidence interval for difference between two means
        4.	Estimate the confidence interval for population proportion(\u03C0) basis the sample proportion(p\u0302)
        """)

    st.subheader("Hypothesis testing")

    with st.expander('Expand to learn about the coverage of hypothesis testing'):
        st.markdown(""" 
        #### Hypothesis Testing
        This part deal with answering question: **If this claim is true for my sample, does it mean I have sufficient evidence for having it true for the population?**
        In this app, I have covered following tests:
        1.	One-sample test for mean when population standard deviation is known
        2.	One-sample test for mean when population standard deviation is not known
        3.	One-sample test for proportion
        4.	Two-samples test for the difference in means of two independent populations
        5.	Two-samples test for the difference in means of two paired populations
        6.	Two-samples test for the difference in proportion of two populations
        7.	ANOVA
        8.	Chi-square test
        """)
elif choice == 'Confidence Interval(CI) Estimation':
    Menu1 = ['CI for {} when {} is unknown'.format(population_mean,population_standard_deviation),'CI for {} basis {}'.format(population_proportion,sample_proportion),'CI for difference between two means',
    'CI for {} when {} is known'.format(population_mean,population_standard_deviation)]
    choice1 = st.sidebar.selectbox('What would you like to do?',Menu1)
    if choice1 == 'CI for {} when {} is known'.format(population_mean,population_standard_deviation):
        st.markdown(""" 
        ##### Estimating the confidence interval for population mean with population standard deviation known
        ###### Assumption made:
                """)
        st.info('If sample size is small(<30), then underlying population distribution is normal')
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        if mode_of_input == 'Individual values':
            with st.expander('Enter the values for estimating the CI'):
                    pop_sd = st.number_input('Enter the population standard deviation')
                    sample_mean = st.number_input('Enter the sample mean')
                    sample_size = st.number_input('Enter the size of the sample')
                    ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
            if pop_sd>0 and sample_mean>0 and sample_size>0 and ci_level>0:
                ll,ul=ci_mu_sigma_known(pop_sd,sample_mean,sample_size,ci_level)
                with st.expander('Results are:'):
                    st.success("One can be {:.0f}% confident that the population mean will lie between {} and {}".format(ci_level*100,np.round(ll,4),np.round(ul,4)))
        else:
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        pop_sd = st.number_input('Enter the population standard deviation')
                        ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                        sample_mean = np.mean(df)[0]
                        sample_size = df.shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Population Standard Deviation':pop_sd,'Sample mean':sample_mean,'Sample size':sample_size,'Confidence Interval':ci_level*100+'%'}
                    ll,ul=ci_mu_sigma_known(pop_sd,sample_mean,sample_size,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population mean will lie between {} and {}".format(ci_level*100,np.round(ll,4),np.round(ul,4)))
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        pop_sd = st.number_input('Enter the population standard deviation')
                        ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                        sample_mean = np.mean(df)[0]
                        sample_size = df.shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Population Standard Deviation':pop_sd,'Sample mean':sample_mean,'Sample size':sample_size,'Confidence Interval':str(ci_level*100)+'%'}
                        st.json(dict1)
                    ll,ul=ci_mu_sigma_known(pop_sd,sample_mean,sample_size,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population mean will lie between {} and {}".format(ci_level*100,np.round(ll,4),np.round(ul,4)))
                else:
                    st.warning("Please upload .csv or .xlsx file only")                    
    elif choice1 == 'CI for {} when {} is unknown'.format(population_mean,population_standard_deviation):
        st.markdown(""" 
        ##### Estimating the confidence interval for population mean with population standard deviation unknown
        ###### Assumption made:
                """)
        st.info('If sample size is small(<30), then underlying population distribution is normal')
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        if mode_of_input == 'Individual values':
            with st.expander('Enter the values for estimating the CI'):
                    sample_sd = st.number_input('Enter the sample standard deviation')
                    sample_mean = st.number_input('Enter the sample mean')
                    sample_size = st.number_input('Enter the size of the sample')
                    ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
            if sample_sd>0 and sample_mean>0 and sample_size>0 and ci_level>0:
                ll,ul=ci_mu_sigma_unknown(sample_sd,sample_mean,sample_size,ci_level)
                with st.expander('Results are:'):
                    st.success("One can be {:.0f}% confident that the population mean will lie between {} and {}".format(ci_level*100,np.round(ll,4),np.round(ul,4)))
        else:
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                        sample_mean = np.mean(df)[0]
                        sample_size = df.shape[0]
                        sample_sd = np.std(df[0])
                    with st.expander('Your input data is'):
                        dict1 = {'Sample Standard Deviation':sample_sd,'Sample mean':sample_mean,'Sample size':sample_size,'Confidence Interval':str(ci_level*100)+'%'}
                        st.json(dict1)
                    ll,ul=ci_mu_sigma_unknown(sample_sd,sample_mean,sample_size,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population mean will lie between {} and {}".format(ci_level*100,np.round(ll,4),np.round(ul,4)))
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                        sample_mean = np.mean(df)[0]
                        sample_size = df.shape[0]
                        sample_sd = np.std(df[0])
                    with st.expander('Your input data is'):
                        dict1 = {'Sample Standard Deviation':sample_sd,'Sample mean':sample_mean,'Sample size':sample_size,'Confidence Interval':str(ci_level*100)+'%'}
                        st.json(dict1)
                    ll,ul=ci_mu_sigma_known(sample_sd,sample_mean,sample_size,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population mean will lie between {} and {}".format(ci_level*100,np.round(ll,4),np.round(ul,4)))
                else:
                    st.warning("Please upload .csv or .xlsx file only") 
    elif choice1 == 'CI for {} basis {}'.format(population_proportion,sample_proportion):
        st.markdown(""" 
        ##### Estimating the confidence interval for population proportion basis sample proportion
                """)
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        if mode_of_input == 'Individual values':
            with st.expander('Enter the values for estimating the CI'):
                checkbox1 = st.checkbox('Would you like to enter sample proportion?')
                if checkbox1:
                    sample_prop = st.number_input('Enter the value of sample proportion')
                else:
                    X = st.number_input('Enter the number of items having the characteristic')
                    sample_size = st.number_input('Enter the size of the sample')
                    ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
            if X>0 and sample_size>0 and ci_level>0:
                ll,ul=ci_prop_pop_sample(X,sample_size,ci_level)
                with st.expander('Results are:'):
                    st.success("One can be {:.0f}% confident that the population proportion will lie between {:.2f}% and {:.2f}%".format(ci_level*100,ll*100,ul*100))
            elif checkbox1==True:
                ll,ul=ci_prop_pop_sample_p_given(sample_prop,ci_level)
                with st.expander('Results are:'):
                    st.success("One can be {:.0f}% confident that the population proportion will lie between {:.2f}% and {:.2f}%".format(ci_level*100,ll*100,ul*100))

        else:
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                        unique_values = tuple(df[0].unique())
                        chosen_unique_value = st.radio('Choose one unique value',unique_values)
                        X = df.loc[df[0]==chosen_unique_value].shape[0]
                        sample_size = df.shape[0]
                        sample_prop = X/sample_size
                    with st.expander('Your input data is'):
                        dict1 = {'Sample Proportion':sample_prop,'Sample size':sample_size,'Confidence Interval':ci_level*100+'%'}
                        st.json(dict1)
                    ll,ul=ci_prop_pop_sample(X,sample_size,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population proportion will lie between {:.2f}% and {:.2f}%".format(ci_level*100,ll*100,ul*100))
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None)
                    ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                    unique_values = tuple(df[0].unique())
                    chosen_unique_value = st.radio('Choose one unique value',unique_values)
                    X = df.loc[df[0]==chosen_unique_value].shape[0]
                    sample_size = df.shape[0]
                    sample_prop = X/sample_size
                    with st.expander('Your input data is'):
                        dict1 = {'Sample Proportion':sample_prop,'Sample size':sample_size,'Confidence Interval':ci_level*100+'%'}
                        st.json(dict1)
                    ll,ul=ci_prop_pop_sample(X,sample_size,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population proportion will lie between {:.2f}% and {:.2f}%".format(ci_level*100,ll*100,ul*100))
                else:
                    st.warning("Please upload .csv or .xlsx file only")
    else:
        st.markdown(""" 
        ##### Estimating the confidence interval for difference between two means
                """)
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        if mode_of_input == 'Individual values':
            with st.expander('Enter the values for estimating the CI'):
                    X1bar = st.number_input('Enter mean for sample 1')
                    X2bar = st.number_input('Enter mean for sample 2')
                    S1 = st.number_input('Enter the standard deviation of the sample 1')
                    S2 = st.number_input('Enter the standard deviation of the sample 2')
                    n1 = st.number_input('Enter the size of the sample 1')
                    n2 = st.number_input('Enter the size of the sample 2')
                    ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
            if X1bar>0 and X2bar>0 and S1>0 and S2>0 and n1>0 and n2>0:
                ll,ul=confidence_interval_mean_difference(X1bar,X2bar,S1,S2,n1,n2,ci_level)
                with st.expander('Results are:'):
                    st.success("One can be {:.0f}% confident that the population proportion will lie between {} and {}".format(ci_level*100,ll,ul))
        else:
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                        X1bar = np.mean(df[0])
                        X2bar = np.mean(df[1])
                        S1 = np.std(df[0])
                        S2 = np.std(df[1])
                        n1 = df.loc[df[0].isna()!=1,0].shape[0]
                        n2 = df.loc[df[1].isna()!=1,1].shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Sample 1 mean':X1bar,'Sample 1 size':n1,'Sample 1 standard deviation':S1,'Sample 2 mean':X2bar,'Sample 2 standard deviation':S2,'Sample 2 size':n2,'Confidence Interval':str(ci_level*100)+'%'}
                        st.json(dict1)
                    ll,ul=confidence_interval_mean_difference(X1bar,X2bar,S1,S2,n1,n2,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population proportion will lie between {} and {}".format(ci_level*100,ll,ul))
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None)
                    ci_level = st.number_input('Enter the confidence interval for which calculation needs to be done',min_value=0.0,max_value=1.0)
                    X1bar = np.mean(df[0])
                    X2bar = np.mean(df[1])
                    S1 = np.std(df[0])
                    S2 = np.std(df[1])
                    n1 = df.loc[df[0].isna()!=1,0].shape[0]
                    n2 = df.loc[df[1].isna()!=1,1].shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Sample 1 mean':X1bar,'Sample 1 size':n1,'Sample 1 standard deviation':S1,'Sample 2 mean':X2bar,'Sample 2 standard deviation':S2,'Sample 2 size':n2,'Confidence Interval':str(ci_level*100)+'%'}
                        st.json(dict1)
                    ll,ul=confidence_interval_mean_difference(X1bar,X2bar,S1,S2,n1,n2,ci_level)
                    with st.expander('Results are:'):
                        st.success("One can be {:.0f}% confident that the population proportion will lie between {} and {}".format(ci_level*100,ll,ul))
                else:
                    st.warning("Please upload .csv or .xlsx file only")

elif choice == 'Hypothesis testing':
    Menu1 = ['One-sample test for {} when {} is known'.format(population_mean,population_standard_deviation),'One-sample test for {} when {} is unknown'.format(population_mean,population_standard_deviation),'One-sample test for proportion',
    'Two-samples test for the difference in means of two independent populations','Two-samples test for the difference in means of two paired populations','Two-samples test for the difference in proportion of two populations','ANOVA','Chi-square test']
    choice1 = st.sidebar.selectbox('What would you like to do?',Menu1)
    if choice1 == 'One-sample test for {} when {} is known'.format(population_mean,population_standard_deviation):
        st.markdown(""" 
        ##### Hypothesis testing for population mean when population standard deviation is known
        ###### Assumption made:
                """)
        st.info('If sample size is small(<30), then underlying population distribution is normal')
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        test_type = st.selectbox('What kind of test it is?',('Left tail test','Right tail test','Two tail test'))
        if mode_of_input == 'Individual values':
            Mu = st.number_input('Enter the population mean')
            sigma = st.number_input('Enter the population standard deviation')
            Xbar = st.number_input('Enter the sample mean')
            n = st.number_input('Enter the size of the sample')
            alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
            if Mu>0 and sigma>0 and Xbar>0 and alpha>0 and n>0:
                z_stat,p_value,Decision=z_test(Xbar,Mu,sigma,n,alpha,test_type)
                with st.expander('Results are:'):
                    st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,3),np.round(p_value,3),Decision))
        
        else:
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        Mu = st.number_input('Enter the population mean')
                        sigma = st.number_input('Enter the population standard deviation')
                        alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
                        Xbar = np.mean(df)[0]
                        n = df.shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Population Mean':Mu,'Population Standard Deviation':sigma,'Sample mean':Xbar,'Sample size':n,'Level of significance':alpha}
                    z_stat,p_value,Decision=z_test(Xbar,Mu,sigma,n,alpha,test_type)
                    with st.expander('Results are:'):
                        st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,3),np.round(p_value,3),Decision))
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None,sheet_name='Sheet1')
                    with st.expander('Enter these values for estimating the CI'):
                        Mu = st.number_input('Enter the population mean')
                        sigma = st.number_input('Enter the population standard deviation')
                        alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
                        Xbar = np.mean(df)[0]
                        n = df.shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Population Mean':Mu,'Population Standard Deviation':sigma,'Sample mean':Xbar,'Sample size':n,'Level of significance':alpha}
                    z_stat,p_value,Decision=z_test(Xbar,Mu,sigma,n,alpha,test_type)
                    with st.expander('Results are:'):
                        st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,3),np.round(p_value,3),Decision))

    elif choice1 == 'One-sample test for {} when {} is unknown'.format(population_mean,population_standard_deviation):
        st.markdown(""" 
        ##### Hypothesis testing for population mean when population standard deviation is known
        ###### Assumption made:
                """)
        st.info('If sample size is small(<30), then underlying population distribution is normal')
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        test_type = st.selectbox('What kind of test it is?',('Left tail test','Right tail test','Two tail test'))
        if mode_of_input == 'Individual values':
            Mu = st.number_input('Enter the population mean')
            Xbar = st.number_input('Enter the sample mean')
            n = st.number_input('Enter the size of the sample')
            S = st.number_input('Enter the sample standard deviation')
            alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
            if Mu>0 and Xbar>0 and alpha>0 and n>0:
                t_stat,p_value,Decision=t_test(Xbar,Mu,S,n,alpha,test_type)
                with st.expander('Results are:'):
                    st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))
        
        else:
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for doing the hypothesis test'):
                        Mu = st.number_input('Enter the population mean')
                        alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
                        Xbar = np.mean(df)[0]
                        n = df.shape[0]
                        S = np.std(df[0])
                    with st.expander('Your input data is'):
                        dict1 = {'Population Mean':Mu,'Sample Standard Deviation':S,'Sample mean':Xbar,'Sample size':n,'Level of significance':alpha}
                    t_stat,p_value,Decision=t_test(Xbar,Mu,S,n,alpha,test_type)
                    with st.expander('Results are:'):
                        st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None,sheet_name='Sheet1')
                    with st.expander('Enter these values for doing the hypothesis test'):
                        Mu = st.number_input('Enter the population mean')
                        alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
                        Xbar = np.mean(df)[0]
                        n = df.shape[0]
                        S = np.std(df[0])
                    with st.expander('Your input data is'):
                        dict1 = {'Population Mean':Mu,'Sample Standard Deviation':S,'Sample mean':Xbar,'Sample size':n,'Level of significance':alpha}
                    t_stat,p_value,Decision=t_test(Xbar,Mu,S,n,alpha,test_type)
                    with st.expander('Results are:'):
                        st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))
    elif choice1 == 'One-sample test for proportion':
        st.markdown(""" 
        ##### Hypothesis testing for one sample proportion
                """)
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        test_type = st.selectbox('What kind of test it is?',('Left tail test','Right tail test','Two tail test'))
        if mode_of_input == 'Individual values':
            checkbox1 = st.checkbox('Would you like to enter sample proportion?')
            if checkbox1:
                sample_prop = st.number_input('Enter the value of sample proportion')
            else:
                count = st.number_input('Enter the number of items having the desired characteristic')
            pop_prop = st.number_input('Enter the population proportion')
            nobs = st.number_input('Enter the sample size')
            alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
            if count>0 and nobs>0 and alpha>0 and pop_prop>0:
                sample_prop = count/nobs
                z_stat,p_value,Decision=one_sample_proportion_ztest(sample_prop,pop_prop,nobs,alpha,test_type)
                with st.expander('Results are:'):
                    st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,5),np.round(p_value,5),Decision))
            elif checkbox1!=False and alpha>0 and pop_prop>0:
                z_stat,p_value,Decision=one_sample_proportion_ztest(sample_prop,pop_prop,nobs,alpha,test_type)
                with st.expander('Results are:'):
                    st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,5),np.round(p_value,5),Decision))
        
        else:
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for estimating the CI'):
                        unique_values = tuple(df[0].unique())
                        chosen_unique_value = st.radio('Choose the value for which proportion needs to be tested',unique_values)
                        count = df.loc[df[0]==chosen_unique_value].shape[0]
                        nobs = df.shape[0]
                        pop_prop = st.number_input('Enter the population proportion')
                        alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
                    with st.expander('Your input data is'):
                        dict1 = {'The number of successes is':count,'The number of trials':nobs,'Hypothesized population proportion':pop_prop,'Level of significance':alpha}
                    if test_type == 'Left tail test':
                        alternative = 'smaller'
                    elif test_type == 'Right tail test':
                        alternative = 'larger'
                    else:
                        alternative = 'two-sided'
                    z_stat,p_value=proportions_ztest(count=count,nobs=nobs,value=pop_prop,alternative=alternative)
                    if alternative == 'two-sided':
                        p_value = p_value*2
                    if p_value>=alpha:
                        Decision = 'Fail to reject the null hypothesis'
                    else:
                        Decision = 'Reject the null hypothesis'
                    with st.expander('Results are:'):
                        st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,5),np.round(p_value,5),Decision))
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None,sheet_name='Sheet1')
                    with st.expander('Enter these values for estimating the CI'):
                        unique_values = tuple(df[0].unique())
                        chosen_unique_value = st.radio('Choose the value for which proportion needs to be tested',unique_values)
                        count = df.loc[df[0]==chosen_unique_value].shape[0]
                        nobs = df.shape[0]
                        pop_prop = st.number_input('Enter the population proportion')
                        alpha = st.number_input('Enter the level of significance',min_value=0.0,max_value=1.0)
                    with st.expander('Your input data is'):
                        dict1 = {'The number of successes is':count,'The number of trials':nobs,'Hypothesized population proportion':pop_prop,'Level of significance':alpha}
                    if test_type == 'Left tail test':
                        alternative = 'smaller'
                    elif test_type == 'Right tail test':
                        alternative = 'larger'
                    else:
                        alternative = 'two-sided'
                    z_stat,p_value=proportions_ztest(count=count,nobs=nobs,value=pop_prop,alternative=alternative)
                    if alternative == 'two-sided':
                        p_value = p_value*2
                    if p_value>=alpha:
                        Decision = 'Fail to reject the null hypothesis'
                    else:
                        Decision = 'Reject the null hypothesis'
                    with st.expander('Results are:'):
                        st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,5),np.round(p_value,5),Decision))
        
    elif choice1 == 'Two-samples test for the difference in means of two independent populations':
        st.markdown(""" 
        ##### Hypothesis testing for difference in means of two independent populations
        ###### Assumption made:
                """)
        st.info('The variances of the two populations are equal')
        tup1 = tuple(('Individual values','Data in excel or csv'))
        mode_of_input = st.radio("How would you like to input the data?",tup1)
        test_type = st.selectbox('What kind of test it is?',('Left tail test','Right tail test','Two tail test'))
        if mode_of_input == 'Individual values':
            hypothesized_mean_difference = st.number_input('Enter the value of Hypothesized difference in the means of the two populations')
            Xbar1 = st.number_input('Enter the mean for sample 1')
            Xbar2 = st.number_input('Enter the mean for sample 2')
            S1 = st.number_input("Enter the standard deviation for sample 1 :")
            S2 = st.number_input("Enter the standard deviation for sample 2 :")
            n1 = st.number_input("What is the size for sample 1?")
            n2 = st.number_input("What is the size for sample 2?")
            alpha = st.number_input("What is the level of significance?")
            if Xbar1>0 and Xbar2>0 and S1>0 and S2>0 and n1>0 and n2>0:
                t_stat,p_value,Decision=pooled_variance_t_test(Xbar1,Xbar2,S1,S2,n1,n2,alpha)
                with st.expander('Results are:'):
                    st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))
        else:
            st.info('Enter the data in the form of excel or csv below')
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,header=None)
                    with st.expander('Enter these values for doing the hypothesis test'):
                        hypothesized_mean_difference = st.number_input('Enter the value of hypothesized mean difference')
                        alpha = st.number_input('Enter the value for level of significance',min_value=0.0,max_value=1.0)
                        X1bar = np.mean(df[0])
                        X2bar = np.mean(df[1])
                        S1 = np.std(df[0])
                        S2 = np.std(df[1])
                        n1 = df.loc[df[0].isna()!=1,0].shape[0]
                        n2 = df.loc[df[1].isna()!=1,1].shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Sample 1 mean':X1bar,'Sample 1 size':n1,'Sample 1 standard deviation':S1,'Sample 2 mean':X2bar,'Sample 2 standard deviation':S2,'Sample 2 size':n2,'Level of significance':alpha}
                        st.json(dict1)
                    with st.expander('Results are :'):
                        if test_type == "Left tail test":
                            alternative = "smaller"
                        elif test_type == "Right tail test":
                            alternative = "larger"
                        else:
                            alternative = 'two-sided'
                        t_stat,p_value,_ = statsmod.ttest_ind(df[0],df[1],alternative=alternative)
                        if p_value>=alpha:
                            Decision = 'Fail to reject the null hypothesis'
                        else:
                            Decision = 'Reject the null hypothesis'
                        st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))
                if ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,header=None,sheet_name='Sheet1')
                    with st.expander('Enter these values for doing the hypothesis test'):
                        hypothesized_mean_difference = st.number_input('Enter the value of hypothesized mean difference')
                        alpha = st.number_input('Enter the value for level of significance',min_value=0.0,max_value=1.0)
                        X1bar = np.mean(df[0])
                        X2bar = np.mean(df[1])
                        S1 = np.std(df[0])
                        S2 = np.std(df[1])
                        n1 = df.loc[df[0].isna()!=1,0].shape[0]
                        n2 = df.loc[df[1].isna()!=1,1].shape[0]
                    with st.expander('Your input data is'):
                        dict1 = {'Sample 1 mean':X1bar,'Sample 1 size':n1,'Sample 1 standard deviation':S1,'Sample 2 mean':X2bar,'Sample 2 standard deviation':S2,'Sample 2 size':n2,'Level of significance':alpha}
                        st.json(dict1)
                    with st.expander('Results are :'):
                        if test_type == "Left tail test":
                            alternative = "smaller"
                        elif test_type == "Right tail test":
                            alternative = "larger"
                        else:
                            alternative = 'two-sided'
                        t_stat,p_value,_ = statsmod.ttest_ind(df[0],df[1],alternative=alternative)
                        if p_value>=alpha:
                            Decision = 'Fail to reject the null hypothesis'
                        else:
                            Decision = 'Reject the null hypothesis'
                        st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))

    elif choice1 == 'Two-samples test for the difference in means of two paired populations':
        st.markdown(""" 
        ##### Hypothesis testing for difference in means of two paired populations
        ###### Assumption made:
                """)
        st.info('The samples are selected from the populations which are normally distributed')
        test_type = st.selectbox('What kind of test it is?',('Left tail test','Right tail test','Two tail test'))
        hypothesized_mean_difference = st.number_input('Enter the value of Hypothesized difference in the means of the two populations')
        alpha = st.number_input("What is the level of significance?")
        st.info('Enter the data in the form of excel or csv below')
        uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
        if uploaded_file is not None:
            ext = validate_file(uploaded_file)
            if ext == '.csv':
                df = pd.read_csv(uploaded_file,header=None)
                D = df[0] - df[1]
                X1bar = np.mean(df[0])
                X2bar = np.mean(df[1])
                S1 = np.std(df[0])
                S2 = np.std(df[1])
                n1 = df.loc[df[0].isna()!=1,0].shape[0]
                n2 = df.loc[df[1].isna()!=1,1].shape[0]
                with st.expander('Your input data is'):
                    dict1 = {'Sample 1 mean':X1bar,'Sample 1 size':n1,'Sample 1 standard deviation':S1,'Sample 2 mean':X2bar,'Sample 2 standard deviation':S2,'Sample 2 size':n2,'Level of significance':alpha}
                    st.json(dict1)
                with st.expander('Results are :'):
                    if test_type == "Left tail test":
                        alternative = "smaller"
                    elif test_type == "Right tail test":
                        alternative = "larger"
                    else:
                        alternative = 'two-sided'
                    D1 = statsmod.DescrStatsW(D)
                    t_stat,p_value,_ = D1.ttest_mean(value=hypothesized_mean_difference,alternative=alternative)
                    if p_value>=alpha:
                        Decision = 'Fail to reject the null hypothesis'
                    else:
                        Decision = 'Reject the null hypothesis'
                    st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))
            if ext == '.xlsx':
                df = pd.read_excel(uploaded_file,header=None,sheet_name='Sheet1')
                D = df[0] - df[1]
                X1bar = np.mean(df[0])
                X2bar = np.mean(df[1])
                S1 = np.std(df[0])
                S2 = np.std(df[1])
                n1 = df.loc[df[0].isna()!=1,0].shape[0]
                n2 = df.loc[df[1].isna()!=1,1].shape[0]
                with st.expander('Your input data is'):
                    dict1 = {'Sample 1 mean':X1bar,'Sample 1 size':n1,'Sample 1 standard deviation':S1,'Sample 2 mean':X2bar,'Sample 2 standard deviation':S2,'Sample 2 size':n2,'Level of significance':alpha}
                    st.json(dict1)
                with st.expander('Results are :'):
                    if test_type == "Left tail test":
                        alternative = "smaller"
                    elif test_type == "Right tail test":
                        alternative = "larger"
                    else:
                        alternative = 'two-sided'
                    D1 = statsmod.DescrStatsW(D)
                    t_stat,p_value,_ = D1.ttest_mean(value=hypothesized_mean_difference,alternative=alternative)
                    if p_value>=alpha:
                        Decision = 'Fail to reject the null hypothesis'
                    else:
                        Decision = 'Reject the null hypothesis'
                    st.success("T-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(t_stat,3),np.round(p_value,3),Decision))

    elif choice1 == 'Two-samples test for the difference in proportion of two populations':
        st.markdown(""" 
        ##### Hypothesis testing for difference in proportions of two populations
                """)
        with st.expander('Enter these values'):
            test_type = st.selectbox('What kind of test it is?',('Left tail test','Right tail test','Two tail test'))
            hypothesized_mean_difference = st.number_input('Enter the value of Hypothesized difference in the means of the two populations')
            alpha = st.number_input("What is the level of significance?")
            checkbox1 = st.checkbox('Would you like to enter sample proportions directly?')
            if checkbox1:
                p1 = st.number_input('Enter the proportion for sample 1')
                p2 = st.number_input('Enter the proportion for sample 2')
                n1 = st.number_input('Enter the size for sample 1',value=1)
                n2 = st.number_input('Enter the size for sample 2',value=1)
                X1 = p1 * n1
                X2 = p2 * n2
            else:
                X1 = st.number_input('Enter the number of items of interest in sample 1',value=2)
                X2 = st.number_input('Enter the number of items of interest in sample 2',value=2)
                n1 = st.number_input('Enter the size for sample 1',value=1)
                n2 = st.number_input('Enter the size for sample 2',value=1)
        with st.expander('Results are :'):
            z_stat,p_value,Decision = difference_in_proportion(X1,X2,n1,n2,hypothesized_mean_difference)
            st.success("Z-stat is {}  \np-value is {}  \nTherefore, {}".format(np.round(z_stat,5),np.round(p_value,5),Decision))

    elif choice1 == 'ANOVA':
        st.markdown(""" 
        ##### ANOVA
        1. Null hypothesis of this test is all the groups have equal means
        2. Alternate hypothesis of this is at least one of the groups have unequal means
        ###### Assumption made:
                """)
        st.info('Distribution of each group is normal and Variance between the groups is same')
        type_of_anova = st.radio("Choose the type of anova",('One-way ANOVA','Two-way ANOVA'))
        if type_of_anova == 'One-way ANOVA':
            st.subheader('One-way ANOVA')
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file)
                    dep_col = st.selectbox('Choose the dependent feature for One-way ANOVA',tuple(df.columns.tolist()))
                    indep_col =  st.selectbox('Choose the independent feature for One-way ANOVA',tuple(df.drop(dep_col,axis=1).columns.tolist()))
                    formula = str(dep_col)+'~'+str(indep_col)
                    model = ols(formula,df).fit()
                    aov_table = anova_lm(model)
                    st.info('This analysis is checking if {} is dependent on {}'.format(dep_col,indep_col))
                    st.table(aov_table)
                    st.info('Using the analysis below you can see which group(s) among {} has/have distinct {}'.format(indep_col,dep_col))
                    mc = MultiComparison(df[dep_col],df[indep_col])
                    result = mc.tukeyhsd()
                    data = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
                    data["reject"] = np.where(data['reject']==True,'Yes','No')
                    st.dataframe(data)
                    st.info('The last column titled "reject" is signifying whether we should reject the idea that the population means of these two groups are unequal')
                    
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file)
                    dep_col = st.selectbox('Choose the dependent feature for One-way ANOVA',tuple(df.columns.tolist()))
                    indep_col =  st.selectbox('Choose the independent feature for One-way ANOVA',tuple(df.drop(dep_col,axis=1).columns.tolist()))
                    formula = str(dep_col)+'~'+str(indep_col)
                    model = ols(formula,df).fit()
                    aov_table = anova_lm(model)
                    st.info('This analysis is checking if {} is dependent on {}'.format(dep_col,indep_col))
                    st.table(aov_table)
                    st.info('Using the analysis below you can see which group(s) among {} has/have distinct {}'.format(indep_col,dep_col))
                    mc = MultiComparison(df[dep_col],df[indep_col])
                    result = mc.tukeyhsd()
                    data = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
                    data["reject"] = np.where(data['reject']==True,'Yes','No')
                    st.dataframe(data)
                    st.info('The last column titled "reject" is signifying whether we should reject the idea that the population means of these two groups are unequal') 
        elif type_of_anova == 'Two-way ANOVA':
            st.subheader('Two-way ANOVA')
            st.info('Two-way ANOVA can be used to assess the main and interaction effects. In this section one can see both such effects on the dependent variable')
            uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file)
                    st.subheader('Main effects')
                    dep_col = st.selectbox('Choose one dependent feature for main effect of two-way ANOVA',tuple(df.columns.tolist()))
                    indep_col =  st.selectbox('Choose one independent feature for main effect of two-way ANOVA',tuple(df.drop(dep_col,axis=1).columns.tolist()))
                    formula = str(dep_col)+'~'+str(indep_col)
                    model = ols(formula,df).fit()
                    aov_table = anova_lm(model)
                    with st.expander('Result of main effect of {} on {}'.format(indep_col,dep_col)):
                        st.info('This analysis is checking if {} is dependent on {}'.format(dep_col,indep_col))
                        st.table(aov_table)
                    with st.expander('Multicomparison'):
                        st.info('Using the analysis below you can see which group(s) among {} has/have distinct {}'.format(indep_col,dep_col))
                        mc = MultiComparison(df[dep_col],df[indep_col])
                        result = mc.tukeyhsd()
                        data = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
                        data["reject"] = np.where(data['reject']==True,'Yes','No')
                        st.dataframe(data)
                        st.write('The last column titled "reject" is signifying whether we should reject the idea that the population means of these two groups are unequal')
                    st.subheader('Main and interaction effects')
                    indep_cols = st.multiselect('Choose two independent features for main and interaction effect of two-way ANOVA',tuple(df.drop(dep_col,axis=1).columns.tolist()))
                    formula = str(dep_col)+'~'+str(indep_cols[0])+'+'+str(indep_cols[1])+'+'+str(indep_cols[0])+':'+str(indep_cols[1])
                    model = ols(formula,df).fit()
                    aov_table = anova_lm(model)
                    with st.expander('Result of main and interaction effect of {} and {} on {}'.format(indep_cols[0],indep_cols[1],dep_col)):
                        st.info('This analysis is checking if {} is dependent on {}'.format(dep_col,indep_col))
                        st.table(aov_table)
                    fig = plt.figure()
                    p=sns.pointplot(x=df[indep_cols[0]],y=df[dep_col],ci=None,hue=df[indep_cols[1]])
                    p.set_title('Plot to assess the interaction effects')
                    st.pyplot(fig)

                else:
                    df = pd.read_excel(uploaded_file)
                    st.subheader('Main effects')
                    dep_col = st.selectbox('Choose one dependent feature for main effect of two-way ANOVA',tuple(df.columns.tolist()))
                    indep_col =  st.selectbox('Choose one independent feature for main effect of two-way ANOVA',tuple(df.drop(dep_col,axis=1).columns.tolist()))
                    formula = str(dep_col)+'~'+str(indep_col)
                    model = ols(formula,df).fit()
                    aov_table = anova_lm(model)
                    with st.expander('Result of main effect of {} on {}'.format(dep_col,indep_col)):
                        st.info('This analysis is checking if {} is dependent on {}'.format(dep_col,indep_col))
                        st.table(aov_table)
                    with st.expander('Multicomparison'):
                        st.info('Using the analysis below you can see which group(s) among {} has/have distinct {}'.format(indep_col,dep_col))
                        mc = MultiComparison(df[dep_col],df[indep_col])
                        result = mc.tukeyhsd()
                        data = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
                        data["reject"] = np.where(data['reject']==True,'Yes','No')
                        st.dataframe(data)
                        st.write('The last column titled "reject" is signifying whether we should reject the idea that the population means of these two groups are unequal')
                    st.subheader('Main and interaction effects')
                    indep_cols = st.multiselect('Choose two independent features for main and interaction effect of two-way ANOVA',tuple(df.drop(dep_col,axis=1).columns.tolist()))
                    formula = str(dep_col)+'~'+str(indep_cols[0])+'+'+str(indep_cols[1])+'+'+str(indep_cols[0])+':'+str(indep_cols[1])
                    model = ols(formula,df).fit()
                    aov_table = anova_lm(model)
                    with st.expander('Result of main and interaction effect of {} and {} on {}'.format(indep_cols[0],indep_cols[1],dep_col)):
                        st.info('This analysis is checking if {} is dependent on {}'.format(dep_col,indep_col))
                        st.table(aov_table)
                    fig = plt.figure()
                    p=sns.pointplot(x=df[indep_cols[0]],y=df[dep_col],ci=None,hue=df[indep_cols[1]])
                    p.set_title('Plot to assess the interaction effects')
                    st.pyplot(fig)

    elif choice1 == 'Chi-square test':
        st.markdown(""" 
        ##### Chi-squared test
        1. Null hypothesis of this test is there is no relationship between the categorical variables
        2. Alternate hypothesis is there is a significant relationship between the features
        ###### Assumption made:
                """)
        st.info('The data in the cells should be frequencies, or counts of cases rather than percentages or some other transformation of the data')
        uploaded_file = st.file_uploader("Upload .csv or .xlsx file only with no header")
        checkbox1 = st.checkbox('Are you uploading the data in the contingency table format already?')
        if checkbox1 is False:
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file)
                    cat_col = df.select_dtypes(exclude = [np.number]).columns.tolist()
                    cat_1 = st.selectbox('Choose the first category feature',tuple(cat_col))
                    cat_col1 = df.drop(cat_1,axis=1).columns.tolist()
                    cat_2 = st.selectbox('Choose the seond category feature',tuple(cat_col1))
                    alpha = st.number_input('Enter the level of significance for the test')
                    df1 = pd.crosstab(df.cat_1,df.cat_2)
                    with st.expander('Result of Chi-squared test'):
                        st.info('This analysis is checking if there is a relationship between {} and {}'.format(cat_1,cat_2))
                        arr = np.array(df1.values)
                        chi_sq_stat,p_value,deg_freedom,exp_freq=stats.chi2_contingency(arr)
                        if p_value>=alpha:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is no relationship between {} and {}'.format(cat_1,cat_2))
                        else:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is a relationship between {} and {}'.format(cat_1,cat_2))
                else:
                    df = pd.read_excel(uploaded_file)
                    cat_col = df.select_dtypes(exclude = [np.number]).columns.tolist()
                    cat_1 = st.selectbox('Choose the first category feature',tuple(cat_col))
                    cat_col1 = df.drop(cat_1,axis=1).columns.tolist()
                    cat_2 = st.selectbox('Choose the seond category feature',tuple(cat_col1))
                    alpha = st.number_input('Enter the level of significance for the test')
                    df1 = pd.crosstab(df.cat_1,df.cat_2)
                    with st.expander('Result of Chi-squared test'):
                        st.info('This analysis is checking if there is a relationship between {} and {}'.format(cat_1,cat_2))
                        arr = np.array(df1.values)
                        chi_sq_stat,p_value,deg_freedom,exp_freq=stats.chi2_contingency(arr)
                        if p_value>=alpha:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is no relationship between {} and {}'.format(cat_1,cat_2))
                        else:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is a relationship between {} and {}'.format(cat_1,cat_2))
        else:
            if uploaded_file is not None:
                ext = validate_file(uploaded_file)
                if ext == '.csv':
                    df = pd.read_csv(uploaded_file,index_col=0)
                    with st.expander('Enter the values for chi-squared test'):
                        cat_1 = st.text_input('Enter the name of categorical feature 1')
                        cat_2 = st.text_input('Enter the name of categorical feature 2')
                        alpha = st.number_input('Enter the level of significance')
                    with st.expander('Results of chi-squared test'):
                        st.info('This analysis is checking if there is a relationship between {} and {}'.format(cat_1,cat_2))
                        chi_sq_stat,p_value,deg_freedom,exp_freq=stats.chi2_contingency(df)
                        if p_value>=alpha:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is no relationship between {} and {}'.format(cat_1,cat_2))

                        else:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is a relationship between {} and {}'.format(cat_1,cat_2))

                
                elif ext == '.xlsx':
                    df = pd.read_excel(uploaded_file,sheet_name='Sheet1',index_col=0)
                    with st.expander('Enter the values for chi-squared test'):
                        cat_1 = st.text_input('Enter the name of categorical feature 1')
                        cat_2 = st.text_input('Enter the name of categorical feature 2')
                        alpha = st.number_input('Enter the level of significance')
                    with st.expander('Results of chi-squared test'):
                        st.info('This analysis is checking if there is a relationship between {} and {}'.format(cat_1,cat_2))
                        chi_sq_stat,p_value,deg_freedom,exp_freq=stats.chi2_contingency(df)
                        if p_value>=alpha:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is no relationship between {} and {}'.format(cat_1,cat_2))
                        else:
                            st.info('Chi-sq stat is {}.  \np-value is {}.'.format(chi_sq_stat,p_value))
                            st.info('There is a relationship between {} and {}'.format(cat_1,cat_2))
                            





































        


            

        




        








        



            



            


















    

            


















