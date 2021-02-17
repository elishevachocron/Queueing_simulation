from Simulator_class import *
import time
import warnings
import socket


warnings.filterwarnings('ignore')
#start = time.time()


def gen_arrivals(arr_rates, mean_patience, total_hours):
    ''' getד as parameters arr_rates, abandonment rate for all kind of customers and produces a dataframe with columns: "arr_num","time","patience", "type"'''
    arrs = pd.DataFrame(columns=["time", "patience", "type"])
    arr_nums = np.random.poisson(arr_rates)
    num_types = len(arr_rates)
    for i in range(num_types):
        for j in range(total_hours):
            arr_dict = {'time': j+np.random.uniform(0, 1, arr_nums[i][j]),
                     'patience': np.random.exponential(mean_patience[i][j], arr_nums[i][j]),
                     'type':[i]*arr_nums[i][j]}
            arrs = arrs.append(pd.DataFrame(arr_dict), ignore_index=True)
            #print(arr_nums[i][j])
            #print(arr_dict)
            #print(arrs)
    arrs = arrs.sort_values(by='time')
    arrs = arrs.reset_index(drop=True)
    arrs.insert(0, "Arrival_num", list(arrs.index))
    return arrs
            


if __name__ == '__main__':

    np.random.seed(42)
    hostname = socket.gethostname() #to know on which computer we are working
    shrunk = input("Shrunk system? : ")  #defining if wou want to generate a reduced or a not reduced system based on the parameters
    for exp in [12]:
        print('Experiment: ' + str(exp))
        if hostname == 'DESKTOP-A925VLR':
            path = r'C:\Users\Elisheva\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_' + str(exp)
        else:
            path = r'C:\Users\elishevaz\Dropbox (BIU)\QueueMining\WorkingDocs\Simulator\Experiments\Experiment_' + str(exp)

        start = time.time()
        df = pd.DataFrame()
        ctypes = 2 #number of custumer types
        ctypes_dict = {0: 'private', 1: 'not_private'} #name of the customers types - how it defines in the parameters files (Arrivals/Service...)
        c_std_dict = {0: 'std_private', 1: 'std_not_private'} #name of the std column in the files parameters
        speed_up_flag = 0 #do we want to add a speed up option - service time sample depending on the number of customer in the queue (see generate_service in the simulator_class)
        total_hours = 12 #number of hours to generate
        if exp == 0 or exp == 12:
            total_hours = 24
        if exp == 2:
            service_distribution = 'Exponential'
        elif exp != 2:
            service_distribution = 'LogNormal'
        elif exp == 6:
            speed_up_flag = 1

        arr_rates_types = np.zeros((2, total_hours), dtype=int) #initialize to 0 the arrival rates parameters for all customers types and for all hour in the day

        #downloading all the parameters files (Matrix of 168 entries containing the parameters info for all hour in a day and for all day in a week)
        df_arrivals = pd.read_csv(path + '/Arrivals.csv')
        df_service_time = pd.read_csv(path + '/Service_time.csv')
        df_abandonment = pd.read_csv(path + '/Abandonment.csv')
        df_number_of_agents = pd.read_csv(path + '/number_of_agents.csv')

        if shrunk == '1':
            '''
            if we choose to reduced the system, we divided the arrival rates and the number of agents by 
            the divisor in order to keep the load of the system.
            In the case we are generating a reduced system (aka:shrink), we extend the simulation to 30 weeks and not 6'''
            divisor = 30
            df_arrivals['private'] = np.ceil(df_arrivals['private'] / divisor).astype(int)
            df_arrivals['not_private'] = np.ceil(df_arrivals['not_private'] / divisor).astype(int)
            df_number_of_agents['number_of_agents'] = np.ceil(df_number_of_agents['number_of_agents']/divisor).astype(int)
            number_of_weeks = np.arange(0, 30, 1)
        else:
            number_of_weeks = np.arange(0, 6, 1)

        for week in number_of_weeks:
            print('---------------------------------Week: ' + str(week) + '---------------------------------')
            for day in [0, 1, 2, 3, 6]: #wihtout saturday
                print('---------------------------------Day: ' + str(day) + '---------------------------------')
                for type in range(ctypes):
                    # arrival rate initialization
                    if exp == 0:
                        #only for exp 0 (resl system) we take the std of the arrival and service rates, as well.
                        #In the rest of the experiments we take only the mean (see: else)
                        mean = df_arrivals[ctypes_dict[type]].loc[df_arrivals['Weekday'] == day].to_numpy()
                        std = df_arrivals[c_std_dict[type]].loc[df_arrivals['Weekday'] == day].to_numpy()
                        arr_rates_types[type] = [int(np.random.uniform(low=max(0, mean[i]-std[i]), high=(mean[i]+std[i]))) for i in range(len(mean))]
                    else:
                        arr_rates_types[type] = df_arrivals[ctypes_dict[type]].loc[df_arrivals['Weekday'] == day].to_numpy()
                num_servers = df_number_of_agents['number_of_agents'].loc[df_number_of_agents['Weekday'] == day].to_numpy()
                max_servers = np.max(num_servers)
                hours = [num_servers > 0]

                #service_time initialization
                if exp == 0: # parameters with std
                    private_mean_s = df_service_time['private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    private_std_s = df_service_time['std_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    service_time_sample_private = np.array([int(np.random.uniform(low=max(0, private_mean_s[i]-private_std_s[i]),
                                                                              high=(private_mean_s[i]+private_std_s[i]))) for i in range(len(private_mean_s))])

                    not_private_mean_s = df_service_time['not_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    not_private_std_s = df_service_time['std_not_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    service_time_sample_not_private = np.array([int(np.random.uniform(low=max(0, not_private_mean_s[i] - not_private_std_s[i]),
                                                                              high=(not_private_mean_s[i] + not_private_std_s[i]))) for i in range(len(not_private_mean_s))])
                else: # parameters without std (mean definition)
                    service_time_sample_not_private = df_service_time['not_private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)
                    service_time_sample_private = df_service_time['private'].loc[df_service_time['Weekday'] == day].to_numpy(dtype=float)

                # from service time to service rate 1/service_time
                service_rates = np.divide(np.full((1, total_hours), 3600, dtype=float), service_time_sample_private)

                # defining a binary matrix (server_schedule) for all server in the system.
                # if a specific server is present at a specific hour, the matrix at this place will be 1, 0 otherwise
                server_schedule = np.reshape([0] * max_servers * total_hours, (max_servers, total_hours))
                for j in range(total_hours):
                    for i in range(max_servers):
                        if i < num_servers[j]:
                            server_schedule[i][j] = 1

                # now call arrivals generator - use arrays "arr_rates_types", "mean_patience_types"
                mean_patience = df_abandonment['wait_time'].loc[df_abandonment['Weekday'] == day].to_numpy()/3600
                #mean_patience_types = np.full((1, ctypes), mean_patience)
                mean_patience_types = [mean_patience, np.array([float('inf')] * total_hours)]

                mean_service_private = service_time_sample_private/3600 #hour unit
                mean_service_not_private = service_time_sample_not_private/3600 #hour unit

                service_mean_types = [mean_service_private, mean_service_not_private]
                #service_std_types = [sigma_private, sigma_not_private]
                arrivals = gen_arrivals(arr_rates_types, mean_patience_types, total_hours)
                # -> produces "arrivals" dataframe with columns: "arr_num","time","patience", "type"
                # then call simulator - use arrays: "arrivals", "server_schedule", "service_mean_types", "service_rates"
                s = Simulator(arrivals, week, day, server_schedule, service_mean_types, service_rates,
                              total_hours=total_hours, max_servers=max_servers, service_distribution=service_distribution,
                              speed_up_flag=speed_up_flag, policy='P_FCFS')
                s.run()
                df = pd.concat([df, s.summary], ignore_index=True)

            df.to_csv(path + '/Sub_simulation_not_FCFS_bis.csv')


        process_time = time.time() - start
        print('Time elapsed: ', int((process_time/60)))
        print(process_time)
        print(process_time/60)
        print('Data Shape: ', df.shape)

        if shrunk == '1':
            df.to_csv(path + '/New_features_simulation_lowest_system.csv')
        else:
            df.to_csv(path+'/Simulation_not_FCFS_bis.csv')


