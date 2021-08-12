#!/local/python/bin/python3
# -*- coding: UTF-8 -*-

# Original Code by Mark Rawlings
# Updates by Steve Mairs.

from __future__ import print_function
import os
import sys
import re
import numpy as np
import pandas as pd

# Import some plotting stuff:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as style


# Import the required SQL library to allow me to pull all the relevant usage
# numbers from the database.
import mysql.connector

# NOTE TO SELF: I WILL NEED TO FIX THE FOLLOWING TO POINT TO A NON-SNAPSHOT VERSION AT SOME POINT...
sys.path.append('python-jcmt_itc_heterodyne/lib')
from jcmt_itc_heterodyne import \
    HeterodyneITC, HeterodyneITCError, HeterodyneReceiver

######## User Input Parameters ##########

# Make the query more flexible, to sleect a min/max date and a date range to avoid (STM)
mindate     = 20200721
maxdate     = 20210729
avoid_range = [20210529,20210709] # 3 DB PADs removed from system, TSYS values high from 20210529 to 20210709
#avoid_range = []
#output_filename = 'Uu_RMS_table_{}_to_{}_with_elaptime.csv'.format(mindate,maxdate) # Should end in '.csv'
output_filename = 'Uu_RMS_table_{}_to_{}_excluding_{}_to_{}_with_elaptime.csv'.format(mindate,maxdate,avoid_range[0],avoid_range[1])
########################################

# Make ITC Object
itc = HeterodyneITC()

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)

# -----

# connecting to the database
dataBase = mysql.connector.connect(
                     host = **WITHHELD**,
                     user = **WITHHELD**,
                     passwd = **WITHHELD**,
                     database = **WITHHELD** )



if len(avoid_range)!=2:

    date_condition = '''
    (c.utdate BETWEEN {} AND {})
    '''.format(mindate,maxdate)

    if len(avoid_range)>0:
        print('\nIgnoring "avoid_range" because its length is not 0 or 2. Querying Dates Between {} and {}, inclusive...\n'.format(mindate,maxdate))
    else:
        print('\nQuerying Dates Between {} and {}...\n'.format(mindate,maxdate))

elif len(avoid_range)==2:
    date_condition = ''' 
    (c.utdate BETWEEN {} AND {} OR c.utdate BETWEEN {} AND {})
    '''.format(mindate,avoid_range[0],avoid_range[1],maxdate)
 
    print('\nQuerying Dates Between {} and {}, excluding {}-{} (dates inclusive)...\n'.format(mindate,maxdate,avoid_range[0],avoid_range[1]))


query1 = '''
select c.*, a.*, o.commentstatus, o.commenttext from jcmt.COMMON as c
JOIN jcmt.ACSIS as a ON c.obsid=a.obsid  LEFT OUTER JOIN omp.ompobslog
as o ON o.obslogid = (SELECT MAX(obslogid) FROM omp.ompobslog AS o2
WHERE o2.obsid=c.obsid) WHERE c.instrume='UU' AND {}
and c.obs_type='science' AND (o.commentstatus=0 OR o.commentstatus is NULL)
'''.format(date_condition)


# Submit the query and return the results as a Pandas dataframe:
df1 = pd.read_sql(query1, dataBase)
print(' FOUND RESULTS! ',len(df1))
print ( df1['commentstatus'])
query2 = '''
select n.* from calibration.noisestats AS n JOIN jcmt.COMMON AS c ON
n.obsid=c.obsid WHERE c.instrume='UU' AND {}
'''.format(date_condition)

# Submit the query and return the results as a Pandas dataframe:
df2 = pd.read_sql(query2, dataBase)

# Since the second query result doesn't contain a separate subsystem number,
#  we need to extract it from the 'file' column that contains the info.
# We do this by using a regex to extract the subsystem number for each
# dataframe row in turn, appending it to subsysnrlist and then wrtiing
# that list into the dataframe as another column, which is set to be of
# the string type to make concatenation easier:

subsysnrlist=[]

for index, row in df2.iterrows():

    fnstr = row['file']
    x = re.match("^(?:[^_]+_){2}([^_ ]+)", fnstr)
    if x:
        ssnr = int(x.groups()[0])
        subsysnrlist.append(ssnr)
df2['subsysnr'] = subsysnrlist
df2['subsysnrstr'] = df2['subsysnr'].astype(str)

# Use the 'obsid' and new 'subsysnr' columns to create a column to allow
# an inner joing of df1 and df2:
df2['obsid_subsysnr'] = df2[['obsid', 'subsysnrstr']].agg('_'.join, axis=1)

# Perform the inner join, resulting in the (rather large) df3 dataframe:
df3 = pd.merge(df1,df2,on='obsid_subsysnr')

# Rename a couple of columns for ease of use later on:
df3.rename(columns = {'subsysnr_x':'subsysnr'}, inplace = True)
df3.rename(columns = {'object':'object_'}, inplace = True)

# =================================

# Display all the df3 column headings, as a sanity check:
#print(list(df3))

# Add in dervied columns for elevation, lofreq, wvmtau, airmass, elapsed time in seconds:
df3['elapsed_time']            = (df3['date_end']-df3['date_obs'])/pd.Timedelta(seconds=1)
df3['elevation']               = (df3['elstart'] + df3['elend']) / 2.
df3['lofreq']                  = (df3['lofreqs'] + df3['lofreqe']) / 2.
df3['wvmtau']                  = (df3['wvmtaust'] + df3['wvmtauen']) / 2.
df3['am']                      = (df3['amstart'] + df3['amend']) / 2.
df3['tautimesairmass']         = df3['wvmtau'] * df3['am']
df3['tautimesairmass_rounded'] = df3['tautimesairmass'].round(decimals=1)

# Add in derived columns for zenith angle, sky frequency
df3['zenith_angle_deg'] = 90. - df3['elevation']
df3['skyfreq'] = df3['restfreq'] / (1.0 + df3['zsource'])

# Add in rounded sky & IF frequencies for later plotting purposes:
df3['skyfreq_rounded'] = df3['skyfreq'].round(decimals=0)
df3['iffreq_rounded'] = df3['iffreq'].round(decimals=1)

# Add in some additional 'NaN' columns to eventually accummdate the various ITC output numbers:
additional_cols = ['itc_rms_int', 'itc_rms_elapsed', 'itc_t_sys', 'itc_lo_freq', 'itc_eta_sky', 'itc_t_rx', 'itc_elapsed_time', 'itc_tau', 'rmsratio_int','rmsratio_elapsed']
for _ in additional_cols:
    df3[_] = float("NaN")

# Create an initially-empty list in which to collect the names of all the science objects observed:
object_list = []

# For each observation in turn, get the parameters used as ITC input:
for index, row in df3.iterrows():

    if "tsmsk" in row['file']:
       continue

    obsid_subsysnr = row['obsid_subsysnr']
    utdate = row['utdate']
    subsysnr = row['subsysnr']
    obs_sb = row['obs_sb']
    iffreq = row['iffreq']
    lofreq = row['lofreq']
    wvmtau = row['wvmtau']
    wvmtaust = row['wvmtaust']
    zenith_angle_deg = row['zenith_angle_deg']
    elevation = row['elevation']
    sw_mode = row['sw_mode']
    int_time = row['int_time']
    elapsed_time = row['elapsed_time']
    zsource = row['zsource']
    skyfreq = row['skyfreq']
    bwmode = row['bwmode']

    if 'MHzx' in bwmode:
       (width,nchan) = bwmode.split('MHzx')
       freq_res = float(width)/float(nchan)

    elif 'GHzx' in bwmode:
       (width,nchan) = bwmode.split('GHzx')
       freq_res = float(width)*1000/float(nchan)
    else:
       print('Unknown ACSIS mode "{}"'.format(bwmode))
       sys.exit(1)

    object_ = row['object_']
    if object_ not in object_list:
        object_list.append(object_)

#  Are "scan" and "raster" really the same?
    if row['sam_mode'] == 'scan' or row['sam_mode'] == 'raster':
       map_mode = HeterodyneITC.RASTER
    elif row['sam_mode'] == 'grid':
       map_mode = HeterodyneITC.GRID
    elif row['sam_mode'] == 'jiggle':
       map_mode = HeterodyneITC.JIGGLE
    else:
        print('Unknown SAM_MODE "{0}"'.format(row['sam_mode']))
        sys.exit(1)

    if row['sw_mode'] == 'pssw':
       sw_mode = HeterodyneITC.PSSW
    elif row['sw_mode'] == 'freqsw':
       sw_mode = HeterodyneITC.FRSW
    elif row['sw_mode'] == 'chop':   #  Does "chop" really equal "BMSW"?
       sw_mode = HeterodyneITC.BMSW
    else:
        print('Unknown SW_MODE "{0}"'.format(row['sw_mode']))
        sys.exit(1)

#  Not sure how x and y relate to height and width, so these may need to
#  be swapped.
    dim_x = row['map_wdth']
    dim_y = row['map_hght']

#  pixel size. This is my best guess....
    dy = row['scan_dy']
    dx = row['scan_vel']*row['steptime']

##  Only way to determine basket weave that I can find is that the msb title
##  sometimes mentions it....
#    basket_weave = ( "basket" in row['msbtitle'] and "weave" in row['msbtitle'] )

#STM - Single observations can't be basket weaves - only the direction of motion matters here for non-square maps
# scannling along the short length is obviously less efficient and therefore higher noise for a given time
    basket_weave = False

#  The ITC prediction seems to be much better for rasters if the integration time is converted
#  from total integration time to integration time per point. Here I assume that Uu has no overscan area.
    if map_mode == HeterodyneITC.RASTER:
        nx = int(dim_x / dx) + 1
        ny = int(dim_y / dy) + 1
        int_time /= nx*ny
        num_points_elapsed_time = None

#  For a jiggle, the number of points cen be obtained from the jiggle name.
    elif map_mode == HeterodyneITC.JIGGLE:
        mt = re.match( r'smu_(\d)x(\d).dat', row['jigl_nam'] )
        if mt:
           nx = int(mt.group(1))
           ny = int(mt.group(2))
           int_time /= nx*ny
           num_points_elapsed_time = nx*ny
    else:
        num_points_elapsed_time = 1

    try:
        (result, extra) = itc.calculate_rms_for_int_time(
            int_time=int_time,  # seconds
            receiver=HeterodyneReceiver.UU,
            map_mode=map_mode,
            sw_mode=sw_mode,
            freq=skyfreq,  # GHz
            freq_res=freq_res,  # MHz
            tau_225=wvmtaust,
            zenith_angle_deg=zenith_angle_deg,
            is_dsb=False,
            dual_polarization=True, # <- Switched from False, as per Graham's recommendation
            n_points=1,  # single pointing assumed <- MAY NEED TO GENERALIZE THIS?
            dim_x=dim_x,
            dim_y=dim_y,
            dx=dx,
            dy=dy,
            basket_weave=basket_weave,
            separate_offs=False,
            continuum_mode=False,
            sideband=obs_sb,
            if_freq=iffreq,
            with_extra_output=True)

        print(index, int_time)
        print('Observation ID & SSNR:', obsid_subsysnr)
        print('Main result: {}'.format(result))
        print('Extra information: {!r}'.format(extra))
#         print(result)
#         print(row)
        df3.loc[index, 'itc_rms_int'] = result

# ITC output format reminder:
# Main result: 0.026065488649759105
# Extra information: {'t_sys': 135.03880498404604, 'lo_freq': 236.5365991943, 'eta_sky': 0.9051646279314096, 't_rx': 61.31763763866746, 'elapsed_time': 526.6, 'tau': 0.08628942240650818}

    except HeterodyneITCError as e:
        print('Error: {}'.format(e))

    try:
        (result, extra) = itc.calculate_rms_for_elapsed_time(
            elapsed_time=elapsed_time,  # seconds
            receiver=HeterodyneReceiver.UU,
            map_mode=map_mode,
            sw_mode=sw_mode,
            freq=skyfreq,  # GHz
            freq_res=freq_res,  # MHz
            tau_225=wvmtaust,
            zenith_angle_deg=zenith_angle_deg,
            is_dsb=False,
            dual_polarization=True, # <- Switched from False, as per Graham's recommendation
            n_points=num_points_elapsed_time,
            dim_x=dim_x,
            dim_y=dim_y,
            dx=dx,
            dy=dy,
            basket_weave=basket_weave,
            separate_offs=False,
            continuum_mode=False,
            sideband=obs_sb,
            if_freq=iffreq,
            with_extra_output=True)

        print(index, elapsed_time)
        print('Observation ID & SSNR:', obsid_subsysnr)
        print('Main result: {}'.format(result))
        print('Extra information: {!r}'.format(extra))
#         print(result)
#         print(row)
        df3.loc[index, 'itc_rms_elapsed'] = result

# ITC output format reminder:
# Main result: 0.026065488649759105
# Extra information: {'t_sys': 135.03880498404604, 'lo_freq': 236.5365991943, 'eta_sky': 0.9051646279314096, 't_rx': 61.31763763866746, 'elapsed_time': 526.6, 'tau': 0.08628942240650818}

    except HeterodyneITCError as e:
        print('Error: {}'.format(e))


df3['rmsratio_int'] = df3['itc_rms_int'] / df3['rms_mean']
df3['rmsratio_elapsed'] = df3['itc_rms_elapsed'] / df3['rms_mean']



# Display useful summary subset of df3, including ITC-derived RMS prediction:
print(df3[['obsid_subsysnr', 'obs_sb', 'object_', 'int_time', 'restfreq', 'molecule', 'rms_mean', 'itc_rms_int', 'itc_rms_elapsed', 'bwmode', 'rmsratio_int','rmsratio_elapsed']])

print('List of unique target objects observed:')
print(object_list)

#print('Columns:')
#print(list(df3))

if output_filename == '':
    df3.to_csv('df3_table.csv', encoding='utf-8')
else:
    df3.to_csv(output_filename, encoding='utf-8')


# Generate an RMS comparison plot, with colors denoting target object:
style.use('seaborn-ticks')
matplotlib.rc('font', family='Times New Roman')

fig1, ax1 = plt.subplots()

ax1.grid(True)
plt.title('Database RMS vs. ITC (int time) Output (colored by target object)')
ax1.set_xlabel('RMS mean (from database)')
ax1.set_ylabel('ITC-generated RMS estimate')

plot_data1_df = df3[['rms_mean', 'itc_rms_int', 'itc_rms_elapsed', 'object_']].copy()
groups = plot_data1_df.groupby("object_")

for name, group in groups:
    plt.plot(group["rms_mean"], group["itc_rms_int"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_int_1.png', bbox_inches='tight', dpi=300)

plt.clf()

fig, ax = plt.subplots()

ax1.grid(True)
plt.title('Database RMS vs. ITC (elapsed time) Output (colored by target object)')
ax1.set_xlabel('RMS mean (from database)')
ax1.set_ylabel('ITC-generated RMS estimate')
for name, group in groups:
    plt.plot(group["rms_mean"], group["itc_rms_elapsed"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_elap_1.png', bbox_inches='tight', dpi=300)
plt.clf()

# Generate an RMS comparison plot, with frequency:
fig2, ax2 = plt.subplots()

ax2.grid(True)
plt.title('ITC RMS (int time) / RMS mean (from database) vs. Sky Frequency [GHz]')
ax2.set_xlabel('Sky frequency [GHz]')
ax2.set_ylabel('ITC RMS / RMS mean (from database)')

plot_data2_df = df3[['skyfreq', 'rmsratio_int', 'rmsratio_elapsed', 'skyfreq_rounded']].copy()
groups = plot_data2_df.groupby("skyfreq_rounded")

for name, group in groups:
    plt.plot(group["skyfreq"], group["rmsratio_int"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_int_2.png', bbox_inches='tight', dpi=300)
plt.clf()

# Generate an RMS comparison plot, with frequency -- ELAPSED TIME:
fig2, ax2 = plt.subplots()

ax2.grid(True)
plt.title('ITC RMS (elap time) / RMS mean (from database) vs. Sky Frequency [GHz]')
ax2.set_xlabel('Sky frequency [GHz]')
ax2.set_ylabel('ITC RMS / RMS mean (from database)')

for name, group in groups:
    plt.plot(group["skyfreq"], group["rmsratio_elapsed"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_elap_2.png', bbox_inches='tight', dpi=300)
plt.clf()


# # Generate an RMS comparison plot, with tau*airmass:
fig3, ax3 = plt.subplots()

ax3.grid(True)
plt.title('ITC RMS (int) / RMS mean (from database) vs. tau*airmass (as a transmission proxy)')
ax3.set_xlabel('tau*airmass, as a transmission proxy)')
ax3.set_ylabel('ITC RMS / RMS mean (from database)')

plot_data3_df = df3[['tautimesairmass', 'rmsratio_int', 'rmsratio_elapsed', 'tautimesairmass_rounded']].copy()
groups = plot_data3_df.groupby("tautimesairmass_rounded")

for name, group in groups:
    plt.plot(group["tautimesairmass"], group["rmsratio_int"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_int_3.png', bbox_inches='tight', dpi=300)
plt.clf()

# # Generate an RMS comparison plot, with tau*airmass - Elapsed time:
fig3, ax3 = plt.subplots()

ax3.grid(True)
plt.title('ITC RMS (elap) / RMS mean (from database) vs. tau*airmass (as a transmission proxy)')
ax3.set_xlabel('tau*airmass, as a transmission proxy)')
ax3.set_ylabel('ITC RMS / RMS mean (from database)')

for name, group in groups:
    plt.plot(group["tautimesairmass"], group["rmsratio_elapsed"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_elap_3.png', bbox_inches='tight', dpi=300)
plt.clf()


# Generate an RMS comparison plot, with IF freq:
fig4, ax4 = plt.subplots()

ax4.grid(True)
plt.title('ITC RMS (int) / RMS mean (from database) vs. IF Frequency [GHz]')
ax4.set_xlabel('IF Freq [GHz]')
ax4.set_ylabel('ITC RMS / RMS mean (from database)')

plot_data4_df = df3[['iffreq', 'rmsratio_int', 'rmsratio_elapsed', 'iffreq_rounded']].copy()
groups = plot_data4_df.groupby("iffreq_rounded")

for name, group in groups:
    plt.plot(group["iffreq"], group["rmsratio_int"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_int_4.png', bbox_inches='tight', dpi=300)
plt.clf()

# Generate an RMS comparison plot, with IF freq -- Elapsed time:
fig4, ax4 = plt.subplots()

ax4.grid(True)
plt.title('ITC RMS (elap) / RMS mean (from database) vs. IF Frequency [GHz]')
ax4.set_xlabel('IF Freq [GHz]')
ax4.set_ylabel('ITC RMS / RMS mean (from database)')

for name, group in groups:
    plt.plot(group["iffreq"], group["rmsratio_elapsed"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_elap_4.png', bbox_inches='tight', dpi=300)
plt.clf()


# Generate an RMS comparison plot, with integration time:
fig5, ax5 = plt.subplots()

ax5.grid(True)
plt.title('ITC RMS (int) / RMS mean (from database) vs. integration time [s]')
ax5.set_xlabel('Integration Time [s]')
ax5.set_ylabel('ITC RMS / RMS mean (from database)')

plot_data5_df = df3[['int_time', 'rmsratio_int']].copy()
groups = plot_data5_df.groupby("int_time")

for name, group in groups:
    plt.plot(group["int_time"], group["rmsratio_int"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_int_5.png', bbox_inches='tight', dpi=300)
plt.clf()

# Generate an RMS comparison plot, with elapsed time:
fig5, ax5 = plt.subplots()

ax5.grid(True)
plt.title('ITC RMS (elap) / RMS mean (from database) vs. integration time [s]')
ax5.set_xlabel('Integration Time [s]')
ax5.set_ylabel('ITC RMS / RMS mean (from database)')

plot_data5_df = df3[['elapsed_time', 'rmsratio_elapsed']].copy()
groups = plot_data5_df.groupby("elapsed_time")

for name, group in groups:
    plt.plot(group["elapsed_time"], group["rmsratio_elapsed"], marker="o", linestyle="", label=name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=1.)

# Generate a PNG image as output:
plt.savefig(fname='Uu_database_vs_ITC_elap_5.png', bbox_inches='tight', dpi=300)
plt.clf()


