import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

def download(url):
    result = None
    with requests.get(url) as response:
        result = response.text
    return result

def save_file(series, filename):
    f = open(filename, 'w')
    f.write(series)
    f.close()

def getDataAsJson(series, readings, sm):
    series = series.split('\n')
    header1 = series[0].split(',')
    header2 = series[1].split(',')
    series = series[2:] # remove the two header lines
    # discover Date field
    dateix = None
    for ix, parameter in enumerate(header1):
        if parameter == 'Date':
            dateix = ix
            break
    data = []
    for ix, parameter in enumerate(header1):
        if parameter.startswith(readings):
            unit = header2[ix]
            parameterdata = {'parameter': parameter, 'unit': unit, 'times': [], 'timeseries': []}
            for line in series:
                line = line.split(',')
                try:
                    time = line[dateix]
                    time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                    value = float(line[ix])
                    parameterdata['timeseries'].append(value)
                    parameterdata['times'].append(time)
                except:
                    pass # null, empty, etc
            data.append(parameterdata)
    # proper sorting, for the figure labels
    if sm: # Soil moisture 5cm, # Soil moiture 10cm, etc.
        data.sort(key=lambda val: float(val['parameter'].split(' ')[-1][:-2]))
    else: # Stem Water Potential probe 1, # Stem Water Potential probe 2
        data.sort(key=lambda val: val['parameter'])
    return data

def plotGraph(data, location):
    plt.grid(visible=True, axis='both', color='#dfdfdf')

    for item in data:
        parameter = item['parameter']
        unit = item['unit']
        values = item['timeseries']
        times = item['times']
        plt.plot(times, values, label=parameter)

    if parameter.startswith('Soil') or parameter.startswith('Stem'):
        readings = ' '.join(parameter.split(' ')[:-1])
    else:
        readings = parameter
    plt.ylabel(readings + ' ' + unit, fontsize=12)
    #plt.xlabel('Date', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(rotation=90, fontsize=12, ha='right')
    plt.title(readings + ' ' + location[1], fontsize=12) #(readings + ' ' + location[0] + ' ' + location[1], fontsize=12)
    legend = plt.legend(loc='upper left', fontsize=10)
    
def getApiUrl(logger_id, mindate, maxdate):
    base_url = 'http://majisysdemo.itc.utwente.nl/florapulse/get7days.py'
    api_url = base_url + '?location=' + logger_id[0] + '&mindate=' + mindate.strftime('%Y-%m-%dT%H_%M_%S') + '&maxdate=' + maxdate.strftime('%Y-%m-%dT%H_%M_%S')
    return api_url

def downloadTimeseries(loggers, mindate, maxdate):
    timeseries = {}
    for ix, logger_id in enumerate(loggers):
        api_url = getApiUrl(logger_id, mindate, maxdate)
        timeseries[logger_id] = download(api_url)
    return timeseries

def groupByParameter(timeseries, parameters, loggers):
    data = {}
    for parameter in parameters:
        for logger_id in loggers:
            d = getDataAsJson(timeseries[logger_id], parameter, parameter.startswith('Soil')) # Soil moisture, Soil temperature etc. (case-sensitive)
            if len(d) > 0:
                if 'timeseries' in d[0]: # probe the first parameter (e.g. Soil Moisture 5cm) and check if there is data in this period
                    if len(d[0]['timeseries']) > 0:
                        if parameter not in data:
                            data[parameter] = []
                        data[parameter].append((logger_id, d))
    return data

def plotData(data, parameter):
    nr_loggers = len(data[parameter])
    fig = plt.figure(figsize=(12, 3*nr_loggers))
    for ix, logger_item in enumerate(data[parameter]):
        logger_id = logger_item[0]
        logger_data = logger_item[1]
        plt.subplot(nr_loggers, 1, ix + 1)
        if ix + 1 < nr_loggers: # disable x-labels (date) for all plots except the last one
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plotGraph(logger_data, logger_id)

def toPandas(data):
    for parameter in data.keys():
        new_data = []
        for ix, logger_item in enumerate(data[parameter]):
            logger_id = logger_item[0]
            logger_data = logger_item[1]
            df_total = None
            for item in logger_data:
                param = item['parameter']
                unit = item['unit']
                values = item['timeseries']
                times = item['times']
                df = pd.DataFrame(values, columns=[param], index=times)
                if df_total is None or (len(df) > 9.0 * len(df_total) / 10.0): #if len(df) > 0: # if dataset (e.g. 20cm) is missing, or has less than 90% of the data, skip it
                    if df_total is None:
                        df_total = df
                    else:
                        df_total = pd.merge(df_total, df, left_index=True, right_index=True) # inner join; discard missing samples as we can't use them to compute SM
            logger_item = (logger_id, unit, df_total)
            new_data.append(logger_item)
        data[parameter] = new_data
    return data

def plotGraphPandas(df_data, location, unit, colors=None):
    plt.grid(visible=True, axis='both', color='#dfdfdf')

    for col in df_data.columns:
        if not colors is None:
            plt.plot(df_data.index, df_data[col], label=col, color=colors[col])
        else:
            plt.plot(df_data.index, df_data[col], label=col)

    if col.startswith('Soil') or col.startswith('Stem'):
        readings = ' '.join(col.split(' ')[:-1])
    else:
        readings = col
    plt.ylabel(readings + ' ' + unit, fontsize=12)
    #plt.xlabel('Date', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(rotation=45, fontsize=12, ha='right')
    plt.title(readings + ' ' + location[1], fontsize=12) #(readings + ' ' + location[0] + ' ' + location[1], fontsize=12)
    legend = plt.legend(loc='upper left', fontsize=10)

def plotDataPandas(data, parameter):
    nr_loggers = len(data[parameter])
    fig = plt.figure(figsize=(12, 3*nr_loggers))
    for ix, logger_item in enumerate(data[parameter]):
        logger_id = logger_item[0]
        unit = logger_item[1]
        logger_data = logger_item[2]
        plt.subplot(nr_loggers, 1, ix + 1)
        if ix + 1 < nr_loggers: # disable x-labels (date) for all plots except the last one
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plotGraphPandas(logger_data, logger_id, unit)

def aggregateMeanPandas(data, interval):
    for parameter in data.keys():
        for ix, logger_item in enumerate(data[parameter]):
            logger_id = logger_item[0]
            unit = logger_item[1]
            logger_data = logger_item[2]
            logger_data = logger_data.resample(interval).mean()
            logger_item = (logger_id, unit, logger_data)
            data[parameter][ix] = logger_item
    return data
