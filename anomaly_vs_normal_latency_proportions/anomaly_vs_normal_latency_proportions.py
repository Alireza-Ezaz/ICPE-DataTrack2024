# Description: This script compares the average proportion of each microservice's latency
# to the total latency across all traces where there is no interference
# with the average proportion of each microservice's latency to the total latency across all traces
# where there is an anomaly injected in one specific microservice.

import matplotlib.pyplot as plt
import pandas as pd
import os


# Load the 'no-interference' data
no_interference_df = pd.read_csv('../tracing-data.tar/tracing-data/social-network/no-interference.csv')

# Additionally calculate the total latency of each trace which is the sum of all microservice latencies
no_interference_df['total_latency'] = no_interference_df.iloc[:, 1:].sum(axis=1)

# Calculate the average proportion of each microservice's latency to the total latency across all traces where there is no interference
no_interference_proportions = no_interference_df.iloc[:, 1:-1].div(no_interference_df['total_latency'], axis=0).mean()

# Set the directory path where all CSV files are located
directory_path = '../tracing-data.tar/tracing-data/social-network/'

# Get a list of CSV files in the specified directory and sort them lexicographically
csv_files = sorted([file for file in os.listdir(directory_path) if file.endswith('.csv')])

# Loop through anomaly files and compare with 'no-interference'
for csv_file in csv_files:
    if csv_file == 'no-interference.csv':
        continue

    df = pd.read_csv(os.path.join(directory_path, csv_file))
    df['total_latency'] = df.iloc[:, 1:].sum(axis=1)
    proportions = df.iloc[:, 1:-1].div(df['total_latency'], axis=0).mean()



    # Creating side-by-side bar plot for each microservice
    plt.figure(figsize=(12, 6))
    short_labels = [label.split('_')[0] for label in proportions.index]
    plt.xticks(ticks=range(len(short_labels)), labels=short_labels)
    plt.bar(x=no_interference_proportions.index, height=no_interference_proportions, width=0.4, align='center', label='No Interference')
    plt.bar(x=proportions.index, height=proportions, width=0.4, align='edge', label=f'Anomaly in {csv_file}')
    plt.xlabel('Microservices')
    plt.ylabel('Average Proportion of Total Latency')
    plt.title(f'Impact of Anomaly on Latency: {csv_file}')
    plt.legend()
    plt.show()