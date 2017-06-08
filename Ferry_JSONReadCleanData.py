
# coding: utf-8

# In[1]:

import json
import pandas as pd
import os
import glob
import time
import numpy as np
import datetime as dt

def MergeFiles(xFilesPath, xOrigData) :
    df    = pd.DataFrame()
    first = True 
    os.chdir(xFilesPath) 
    for infile_name in glob.glob('*.json'): 
        with open(infile_name) as infile:
            try:
                d               = json.load(infile)
                df              = pd.DataFrame(d["vessellist"])
                df['timestamp'] = d['timestamp']
                if first:
                    df.to_csv(xOrigData, mode='w', header=True, index=False)
                    first = False
                else:
                    df.to_csv(xOrigData, mode='a', header=False, index=False)
            except:
                print("Problem on the file:" ,infile_name)
                continue
    
def CleanOrigData(xOrigData, xAllTrip, xDockLatLon):
    print('Data Cleaning start',' Time:', time.time() - start)
    df = pd.read_csv(xOrigData,header=0)
    print('Original Total Records df.shape[0]',  df.shape[0] , ' Time:', time.time() - start) 
    df = df[ df.inservice              == True     ]   # if vessel in service
    df = df[ df.headtxt               != "Stopped" ]   # if vessele stopped waiting
    df = df[ df.aterm_abbrev.isnull() == False     ]   # transferring vessle in another dock
    df = df[ df.aterm_abbrev          != ''        ]   # transferring vessle in another dock
    print('After Filter for (InService,Stopped,abbrev.snull) Total Records df.shape[0]',  df.shape[0], ' Time:', time.time() - start) 
    
    df['timestamp']  = pd.to_datetime(df['timestamp'],                                                               format="%m/%d/%Y %I:%M:%S   %p" )
    df['n_datetime'] = pd.to_datetime(df['timestamp'].dt.year.astype(str) + '/' + df['datetime'].astype(str),        format="%Y/%m/%d   %H:%M")  # Use n_leftdock Date info for n_datetime
    df['n_leftdock'] = pd.to_datetime(df['n_datetime'].dt.date.astype(str)+ ' ' + df['leftdock']+df['leftdockAMPM'], format="%Y-%m-%d %I:%M%p")
    df['n_time_past']= ( ( df['n_datetime'] - df['n_leftdock'] ) / np.timedelta64(1, 'm')).astype(int)
    df               = df[ df.n_time_past != -1 ]                                                                          # For leftdock earlier than datetime problem; trip n_datetime 1 minute earlier than n_leftdock
    df.loc[df.n_leftdock > df.n_datetime, 'n_leftdock'] = df.n_leftdock - np.timedelta64(1,'D')                            # For trip start before midnight and continue next day n_leftdock calc is wrong; to correct it
    df['n_time_past']= ( df['n_datetime'] - df['n_leftdock'] ).astype('timedelta64[m]').astype(int) + 1                    # Now we calculate again the past time, because same n_leftdock info changed , Add 1 MIN, cause we don't know starting time in second
    df['n_time_trip']= df.groupby(['vesselID','n_leftdock','lastdock_id'])['n_time_past'].transform(max) + 1               # Add 1 MIN, cause after last record Ferry is moving to Aterm dock
    
    df['n_arrive']= df.groupby(['vesselID','n_leftdock','lastdock_id'])['n_datetime'].transform(max)+np.timedelta64(1,'m')  # Ferry arrival time columns for all rows
    df = df[ df.n_time_trip > 2 ]

    print('After Filter for (TripTime=0) Total Records df.shape[0]',  df.shape[0],' Time:', time.time() - start) 
    df['n_time_left']    = ( df['n_time_trip'] - df['n_time_past'] )
############ Put lastdock_LatLon 'LDlat, LDlon' , and aterm_latlon 'ATlat, ATlon'  information to the records for distance calculations
    xdf_dock = pd.read_csv(xDockLatLon,header=0)                                   # Read the dock lat,lon info
    for ind, row in xdf_dock.iterrows():
        df.loc[ (df['lastdock_abbrev']== row['lastdock_abbrev']), 'LD_lat' ] = row['lastdock_lat']
        df.loc[ (df['lastdock_abbrev']== row['lastdock_abbrev']), 'LD_lon' ] = row['lastdock_lon']
        df.loc[ (df['aterm_abbrev']   == row['aterm_abbrev']   ), 'AT_lat' ] = row['aterm_lat']   
        df.loc[ (df['aterm_abbrev']   == row['aterm_abbrev']   ), 'AT_lon' ] = row['aterm_lon']       
############ Add Previous TimePast to current record
    df.sort_values(['vesselID', 'n_datetime'], ascending=[1,1], inplace=True )
    df['p_time_past'] = df.groupby(['vesselID', 'n_leftdock', 'lastdock_id'])['n_time_past'].apply(lambda x: x.shift(1) )
    df['p_time_past'].fillna(value=0, inplace=True )    
############ Add Previous lat,lon to the current  record
    df.sort_values(['vesselID', 'n_datetime'], ascending=[1,1], inplace=True )
    df['p_lat'] = df.groupby(['vesselID', 'n_leftdock', 'lastdock_id'])['lat'].apply(lambda x: x.shift(1) )
    df['p_lon'] = df.groupby(['vesselID', 'n_leftdock', 'lastdock_id'])['lon'].apply(lambda x: x.shift(1) )
############ TRIP first record location's previous lat,lon is the LastDock Lat,Lon add this to the Trip first record
    df.p_lat.fillna(df.LD_lat, inplace=True)
    df.p_lon.fillna(df.LD_lon, inplace=True)
############ Ferry distance(meter) calculation from current loc to prev location
    df['n_dist_prevloc'] = 6371 * 1000 * 2 * np.arcsin(
        np.sqrt(
            np.sin((np.radians(df['p_lat']) - np.radians(df['lat']))/2)**2 
            + np.cos(np.radians(df['lat'])) 
            * np.cos(np.radians(df['p_lat'])) 
            * np.sin( (np.radians(df['p_lon']) - np.radians(df['lon']))/2)**2
                ))
############ Ferry's distance from last dock CUMULATIVE
    df.sort_values(['vesselID', 'n_datetime'], ascending=[1,1], inplace=True )
    df['n_dist_past'] = df.groupby(['vesselID', 'n_leftdock']).n_dist_prevloc.cumsum()
    df['n_distTot']   = df.groupby(['vesselID', 'n_leftdock'])['n_dist_past'].transform(max)

    df['DistAterm'] = 6371 * 1000 * 2 * np.arcsin(
        np.sqrt(
            np.sin((  np.radians( df['AT_lat'] ) - np.radians( df['lat'] ))/2)**2 
            + np.cos( np.radians( df['lat']    )) 
            * np.cos( np.radians( df['AT_lat'] )) 
            * np.sin((np.radians( df['AT_lon'] ) - np.radians( df['lon'] ))/2)**2 ))
    df['DistLastdock'] = 6371 * 1000 * 2 * np.arcsin(
        np.sqrt(
            np.sin((  np.radians( df['LD_lat'] ) - np.radians( df['lat'] ))/2)**2 
            + np.cos( np.radians( df['lat']    )) 
            * np.cos( np.radians( df['LD_lat'] )) 
            * np.sin((np.radians( df['LD_lon'] ) - np.radians( df['lon'] ))/2)**2 ))
    df['DistTripLastLocTOAterm']   = df.groupby(['vesselID', 'n_leftdock'])['DistAterm'].transform('last')
############ REMOVE whole Trip records which has last location to Aterm dock distance greater than 1000 meter (Last loc we are useing as a arrival time)
############ Trip could be in wrong route (many sample we have)
    print('Before remove (Trip end location - Aterm dock distance) > 1000m Trips  df.shape[0]',  df.shape[0], ' Time:', time.time() - start)      
    zdf = df[df.DistTripLastLocTOAterm >= 1000]
############ Save file those will be removed 
    zdf.to_csv('/Users/ekinezgi/Documents/FerryProject/tmp/DistTripLastLocTOAterm_over1000m.csv', mode='w', header=True, index=False)
    df = df[ df.DistTripLastLocTOAterm < 1000 ]  
    print('After  remove (Trip end location - Aterm dock distance) > 1000m Trips  df.shape[0]',  df.shape[0], ' Time:', time.time() - start)
############ Generate All Trips Summary info
    df.sort_values(  ['lastdock_abbrev','aterm_abbrev','n_leftdock','name'] , ascending=[1,1,1,1], inplace=True ) 
    ydf = df.groupby(['lastdock_abbrev','aterm_abbrev','n_leftdock','name']).agg({
                                'lat'        :{'Flat':'first','Llat':'last' },
                                'lon'        :{'Flon':'first','Llon':'last' },
                                'name'       :{'Name':'first'},
                                'n_leftdock' :{'Record':'count'  },
                                'n_distTot'  :{'Distance':'first'},
                                'n_time_trip':{'TripDur':'first' },
                                'n_arrive'   :{'Arrive':'first'  },
                                'speed'      :{'SpeedAvrg':'mean'  }
                                }) 
    ydf.columns = ydf.columns.droplevel()    # drop a level from a multi-level column index
    ydf.reset_index(inplace=True)            # Turn Multi-index into column
    ydf.to_csv(xAllTrip, mode='w', header=True, index=False)
############ Create TimeStampInSec time in the second of the day timestamp and add next TimeStamp to the current location record
    df['TimeStampInSec'] = df.timestamp.dt.hour.astype(int)*3600 + df.timestamp.dt.minute.astype(int)*60 + df.timestamp.dt.second.astype(int)
    df.sort_values(['vesselID', 'n_datetime'], ascending=[1,1], inplace=True )
    df['p_TimeStampInSec'] = df.groupby(['vesselID', 'n_leftdock', 'lastdock_id'])['TimeStampInSec'].apply(lambda x: x.shift(1) )
    df.p_TimeStampInSec.fillna( df.TimeStampInSec, inplace=True )
############ Trip Count - Time Average and Latency
    df['n_Count']  = df.groupby(['vesselID', 'n_leftdock']).speed.cumcount()+1
    df['TripAvg']  = df.groupby(['lastdock_abbrev', 'aterm_abbrev'])['n_time_trip'].transform('mean').round()
    df['Latency']  = df['n_time_trip'] - df['TripAvg']
############ Trip Speed Average
    df.sort_values(['vesselID','n_leftdock','n_datetime'], ascending=[1,1,1], inplace=True )
    df['SpeedAvg']       = df.groupby(['lastdock_abbrev', 'aterm_abbrev'])['speed'].transform('mean')
    df['n_SpeedCumsum']  = df.groupby(['vesselID', 'n_leftdock']).speed.cumsum() 
    df['n_SpeedCumAvg']  = df['n_SpeedCumsum'] / df['n_Count']
    df['n_SpeedTripAvg'] = df.groupby(['vesselID', 'n_leftdock'])['speed'].transform('mean')
############ WSF own ETA etaArrival etaDuration and etaTimeLeft
    df.loc[df.eta == 'Calculating', 'eta'] = np.NaN    
    df.sort_values(['vesselID', 'n_leftdock', 'lastdock_id','n_datetime'], ascending=[1,1,1,0], inplace=True )
    df['eta']         = df.groupby(['vesselID', 'n_leftdock', 'lastdock_id'])['eta'    ].fillna(method='ffill')
    df['etaAMPM']     = df.groupby(['vesselID', 'n_leftdock', 'lastdock_id'])['etaAMPM'].fillna(method='ffill')
    df['etaArrival']  = pd.to_datetime(df['n_arrive'].dt.date.astype(str) + " " + df['eta'] + df['etaAMPM'], format="%Y-%m-%d %I:%M%p")
    ##### For ETA arrival after midnight
    df.loc[df.etaArrival > df.n_arrive + np.timedelta64(1380,'m')  , 'etaArrival'] = df.etaArrival - np.timedelta64(1,'D')       # For trip start before midnight and continue next day n_leftdock calc is wrong; to correct it
    df['etaTripDur']  = ( df['etaArrival'] - df['n_leftdock'] ).astype('timedelta64[m]').astype(int) 
    df['etaTimeLeft'] = df['etaTripDur'] - df['n_time_past'] 
############ Running-Cummulative Standard Deviation
    df['RunStdDev']  =  df.groupby(['n_leftdock'])['speed'].apply(pd.expanding_std)
    df['RunStdDev'].fillna(value=0, inplace=True )
    return df



def RemoveProblemTrips(xdf, xStartD, xEndD):    
    print('Before Remove Problem Trips xdf.shape[0]', xdf.shape[0], ' Time:', time.time() - start)
    Remove_df = xdf[ (xdf.n_leftdock < xStartD ) | (xdf.n_leftdock > xEndD ) ]
    Remove_df.to_csv('/Users/ekinezgi/Documents/FerryProject/tmp/DateFilterRemovedRecords.csv', mode='w', header=True, index=False)
    xdf = xdf[ (xdf.n_leftdock >= xStartD ) & (xdf.n_leftdock <= xEndD ) ]
    print('After Date Filter xdf.shape[0]', xdf.shape[0], ' Time:', time.time() - start)
    xFilter = [['FRH','SID','2017-01-02 09:55:00'],['FRH','SID','2017-01-02 09:55:00'],['SID','FRH','2017-01-07 12:04:00'], 
               ['LOP','SHI','2017-02-08 21:55:00'],['LOP','SHI','2017-02-18 20:27:00'],['LOP','SHI','2017-02-23 20:24:00'],
               ['LOP','SHI','2017-02-08 16:00:00'],['LOP','SHI','2017-02-23 13:53:00'],['ORI','LOP','2017-02-13 13:25:00'], 
               ['SOU','FAU','2017-01-26 01:41:00'],['SOU','FAU','2017-03-12 23:06:00'],['SOU','FAU','2017-01-17 23:06:00'],
               ['SOU','FAU','2017-01-02 23:07:00'],['SOU','FAU','2017-01-19 00:26:00'],['SOU','FAU','2017-02-02 17:53:00'],
               ['SOU','FAU','2017-03-12 01:41:00'],['SHI','ORI','2017-02-10 16:01:00'],['PTD','TAH','2017-02-12 21:31:00'],
               ['SHI','ORI','2017-02-14 07:00:00'],['SHI','ORI','2017-02-19 18:28:00'],['SHI','ORI','2017-01-24 12:20:00'],
               ['SHI','ORI','2017-03-08 07:00:00'],['FRH','SHI','2017-01-09 06:17:00'] ] 
    xdf = xdf[ ~((xdf.lastdock_abbrev == 'ORI') & (xdf.aterm_abbrev == 'SHI') ) ]   # 2.
    for val in xFilter:
        xdf = xdf[ ~((xdf.lastdock_abbrev == val[0]) & (xdf.aterm_abbrev == val[1]) &  
                 (xdf.n_leftdock == pd.to_datetime( val[2] ))) ]
    print('After  Removed Probled Trips xdf.shape[0]', xdf.shape[0], ' Time:', time.time() - start)
    return xdf

def CreateFerryData(xdf, xCleanData):   
#### Create Dataframe with only selected variables
    ydf = xdf[['aterm','aterm_abbrev','aterm_id','datetime','departDelayed','eta','etaAMPM','etaBasis',
        'id','inservice','lastdock','lastdock_abbrev','lastdock_id','lat','lon','leftdock',
        'leftdockAMPM','name','nextdep','nextdepAMPM','route','speed','system','vesselID','timestamp',
        'h','head','headtxt','icon','label','mmsi','old','pos','w','xOffSet','yOffSet',
        'n_datetime','n_leftdock','n_time_past','n_arrive','n_time_trip',
        'n_time_left','p_lat','p_lon','n_dist_prevloc','n_dist_past','n_distTot',
        'TimeStampInSec', 'DistAterm','DistTripLastLocTOAterm','AT_lat','AT_lon','DistLastdock',
        'p_time_past','p_TimeStampInSec','TripAvg','Latency','n_Count',
        'etaArrival','etaTripDur','etaTimeLeft','RunStdDev','n_SpeedCumAvg','n_SpeedTripAvg','SpeedAvg']]    
    ydf.to_csv(xCleanData, mode='w', header=True, index=False)

def run():
    vDockLatLon = '/Users/ekinezgi/Documents/FerryProject/DockLatLon.csv'

# 2017Jan01 - 2017Mar17
    vStartD = pd.to_datetime('2017-01-01')
    vEndD   = pd.to_datetime('2017-03-18')
    vFilesPath  = '/Users/ekinezgi/Downloads/FerryData2017Jan01TOMar17/'
    vOrigData   = '/Users/ekinezgi/Documents/FerryProject/2017JanMar_Orig.csv'
    vCleanData  = '/Users/ekinezgi/Documents/FerryProject/2017JanMar_Data.csv'
    vAllTrip   = '/Users/ekinezgi/Documents/FerryProject/2017JanMar_Trip.csv'


#     MergeFiles(vFilesPath, vOrigData)                    # Read vessel json files and append to orig file
    dfx = CleanOrigData(vOrigData, vAllTrip, vDockLatLon)  # Clean vessel orig file and save to vCleanData
    dfy = RemoveProblemTrips(dfx, vStartD, vEndD)
    CreateFerryData(dfy, vCleanData)
    
if __name__ == '__main__':
    start = time.time()
    print('Start Time :', time.ctime())
    run()
    
    end = time.time()
    tot_time = end - start
    print('Total time used :', tot_time)  


# In[ ]:



