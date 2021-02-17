import math
import numpy as np
import pandas as pd


class Simulator():

    def __init__(self, arrivals, week, day, server_schedule, service_mean_types, server_rates, total_hours, max_servers,
                 service_distribution, speed_up_flag, policy='FCFS'):
        '''
        In this function, the intrinsic parameters of the system are set (shifts/parameters)
        '''

        self.week = week
        self.day = day
        # arrivals
        self.arrivals = arrivals
        self.num_arrivals = len(self.arrivals)
        # number of customer types
        self.cust_types = len(service_mean_types)  # = number of different customer types
        # servers schedule and service rates (the latter by types)
        self.service_distribution = service_distribution
        self.speed_up_flag = speed_up_flag
        self.policy = policy
        self.server_schedule = pd.DataFrame(server_schedule, columns=range(total_hours), index=range(max_servers))
        self.server_schedule[total_hours] = self.server_schedule[total_hours - 1]
        self.max_servers = max_servers
        self.num_servers = np.sum(self.server_schedule, axis=0)  # number of available servers each hour
        self.server_rates = server_rates  # average server_rates per hour (24 array)
        # self.service_mean_types = service_mean_types # average service times for each type of customer (array 3*1)
        self.service_mean_types = [np.append(x, x[-1]) for x in service_mean_types]
        # server status
        # Status = 0 (unavailable), 1 (idle), 2 (busy)
        self.server_status = pd.DataFrame(columns=['Status', 'Arr_num', 'Start', 'Finish', 'Wait', 'Type'],
                                          index=range(self.max_servers))
        self.hour = ((np.sum(self.server_schedule, axis=0)) > 0).idxmax()  # when servers/customers first arrive
        # server status initialization in that hour
        self.available = self.num_servers[self.hour]
        self.server_status['Status'] = [0] * self.max_servers
        self.server_status.loc[range(self.available), 'Status'] = [
                                                                      1] * self.available  # w.l.o.g. lower index servers are available
        self.server_status.loc[:, 'Start'] = [-float('inf')] * self.max_servers
        self.server_status.loc[:, 'Finish'] = [float('inf')] * self.max_servers
        # initializing queue and numbers in queue (one for each type)
        self.queue = []
        self.in_queue = [0] * self.cust_types
        # prepapring "self.summary" output df, and initializing it
        cols = ['Arrival_num', 'Arrival_time', 'Checkin_time', 'Service_time', 'Exit_time', 'type', 'n_servers',
                'n_available', 'n_not_available', 'LSD', 'Real_WT']
        for i in range(self.cust_types):
            cols.extend(['queue' + str(i + 1)])
        self.summary = pd.DataFrame(columns=cols, index=range(self.num_arrivals))
        self.summary.loc[:, 'Arrival_num'] = self.arrivals['Arrival_num']
        self.summary.loc[:, 'Arrival_time'] = self.arrivals['time']
        self.summary.loc[:, 'Exit_time'] = float('inf') * self.num_arrivals
        # initializing lists of times of arrivals, abandons, departures
        self.list_t_arrival = list(self.arrivals['time'])
        self.list_t_abandon = list(self.arrivals['time'] + self.arrivals['patience'])
        self.list_t_depart = list(self.server_status['Finish'])
        # initializing the clock
        self.clock = 0.0 + self.hour
        self.max_time = float(total_hours)
        # print(cols)
        # print(self.summary)

    def run(self):
        '''
        This function triggers the simulator execution.
        It continuously calls the function “advance_time” (described below),
        until the simulator’s clock reached the execution time set previously
        by the user through the “Init” function.
        '''

        while (min(self.list_t_arrival) < self.max_time) or (min(self.list_t_depart) < self.max_time) or not np.all(
                (np.array(self.in_queue) == 0)):
            # while (min(self.list_t_arrival) < self.max_time) or (min(self.list_t_depart) < 50):
            self.advance_time()
            # print(self.clock)

        # Global variables - correct for all the lines in summary
        self.summary['Week'] = [self.week] * self.num_arrivals
        self.summary['Day'] = [self.day] * self.num_arrivals
        # self.summary['Real_WT'] = self.summary['Checkin_time'] - self.summary['Arrival_time']

    def advance_time(self):
        '''
        This method is the clock of the system.
        It checks the next different events to come. 4 types of events may occur:
        -	Treating the arrival of a customer
        -	Treating the departure of a customer (after getting service)
        -   Treating the abandonment event
        -	Treating a modification in the new hour shift
        Once the event to handle is selected (according to several conditions), the clock value is modified to be equal to the event time, and the proper
        function to handle the event is called.
        '''
        self.t_next_hour = self.hour + 1
        self.t_depart = min(self.list_t_depart)

        self.t_abandon = min(self.list_t_abandon)
        self.t_arrival = min(self.list_t_arrival)

        t_event = min(self.t_arrival, self.t_depart, self.t_abandon, self.t_next_hour)
        self.clock = t_event
        if self.t_depart == t_event:
            self.handle_depart_event()
        elif self.t_arrival == t_event:
            self.handle_arrival_event()
        elif self.t_next_hour == t_event:
            self.handle_change_hour_event()
        elif self.t_abandon == t_event:
            self.handle_abandon_event()

    def handle_arrival_event(self):
        '''
        When called, this method deals with a new arrival event: a new customer enters the system.
        The method adds directly the available information for the newly arrived customer: arrival time, n_available....
        It further generates its service time, drawn from an log normal distribution with a parameter mu defined by the
        user during the initialization step.
        Following this step, the method checks whether a server is available. Two options may occur:
        -   A server is available: in this case, the missing information in the DataFrame may be completed
        as we know the exit time of the customer and his waiting time (his waiting time equals 0 and the
        service time has been generated).
        -   All the servers are used: the customer is added to a waiting queue.
        He will be dequeued when he reaches the top of the queue and one of the servers becomes available.
        At this step the remaining information remains unknown.
        '''
        self.arrival_num = np.argmin(self.list_t_arrival)
        self.list_t_arrival[self.arrival_num] = float('inf')
        self.summary.loc[self.arrival_num, 'Arrival_time'] = self.clock
        self.summary.loc[self.arrival_num, 'n_available'] = len(self.server_status.loc[self.server_status.Status == 1])
        self.summary.loc[self.arrival_num, 'n_not_available'] = len(
            self.server_status.loc[self.server_status.Status == 2])
        self.summary.loc[self.arrival_num, 'n_servers'] = self.num_servers[self.hour]
        self.summary.loc[self.arrival_num, 'type'] = self.arrivals['type'].iloc[self.arrival_num]

        #
        inds = np.array((self.server_status.iloc[:, 0] == 1)) & np.array(
            (self.server_schedule.iloc[:, self.hour] > 0))  # test for idle

        if np.sum(inds) == 0:  # no available server and therefore enters queue
            self.summary.iloc[self.arrival_num, - self.cust_types:] = self.in_queue
            if np.where(((self.summary.Service_time.notna()) & ((self.summary.Exit_time < self.clock))))[0].size != 0:
                lsd_idx = self.summary.index[
                    self.summary['Exit_time'] == max(self.summary.iloc[np.where(
                        ((self.summary.Service_time.notna()) & ((self.summary.Exit_time < self.clock))))[0]][
                                                         'Exit_time'])]
                self.summary.loc[self.arrival_num, ['LSD']] = float(self.summary.loc[lsd_idx[-1], 'Service_time'])
            else:
                self.summary.loc[self.arrival_num, ['LSD']] = 0
            # update queue information
            if self.policy == 'FCFS' or self.arrivals.type[self.arrival_num] == 0 or not self.queue:
                self.queue.append([self.arrival_num, self.arrivals.type[self.arrival_num]])
                self.in_queue[self.queue[-1][1]] += 1
            else:
                idx = np.sum(self.queue, axis=0)[1]
                self.queue = np.insert(self.queue, 2 * idx,
                                       [self.arrival_num, self.arrivals.type[self.arrival_num]]).reshape(-1, 2).tolist()
                self.in_queue[1] += 1


        else:  # goes straight into service
            server_num = np.argmax(inds)
            self.server_status.loc[server_num, 'Status'] = 2  # becomes busy
            self.server_status.loc[server_num, 'Arr_num'] = self.arrival_num
            self.server_status.loc[server_num, 'Start'] = self.t_arrival
            self.customer_type = self.arrivals.loc[self.arrival_num, 'type']
            service_time = self.generate_service()  # use type and possible other state variables
            self.server_status.loc[server_num, 'Finish'] = self.t_arrival + service_time
            self.server_status.loc[server_num, 'Wait'] = 0
            self.server_status.loc[server_num, 'Type'] = self.customer_type
            self.summary.loc[self.arrival_num, ['Checkin_time', 'Service_time', 'Exit_time', 'Real_WT']] = \
                [self.clock, service_time, self.clock + service_time, 0]
            self.summary.loc[self.arrival_num, 'LSD'] = None
            self.summary.iloc[self.arrival_num, -self.cust_types:] = [0, 0]
            if np.where(((self.summary.Service_time.notna()) & ((self.summary.Exit_time < self.clock))))[0].size != 0:
                lsd_idx = self.summary.index[
                    self.summary['Exit_time'] == max(self.summary.iloc[np.where(((self.summary.Service_time.notna()) \
                                                                                 & ((
                                        self.summary.Exit_time < self.clock))))[0]]['Exit_time'])]
                self.summary.loc[self.arrival_num, ['LSD']] = float(self.summary.loc[lsd_idx[-1], 'Service_time'])

            else:
                self.summary.loc[self.arrival_num, ['LSD']] = 0
            self.list_t_abandon[self.arrival_num] = float('inf')
            self.list_t_depart = list(self.server_status['Finish'])

    ##
    def handle_depart_event(self):
        '''
        When called, this method handles the departure of a customer (from his service).
        In this case the specific server that treated the previous customer is released. If the queue is not empty, the
        head of line is dequeued, a service time for this customer is generated, and the missing information
        regarding this customer is completed.
        '''
        server_idx = np.argmin(list(self.server_status['Finish']))
        self.server_status.iloc[server_idx, :] = [1, None, None, float('inf'), None, None]

        if self.server_schedule.iloc[server_idx, self.hour] == 0:
            # self.server_status.loc[server_idx, 'Status'] = 0
            self.server_status.loc[server_idx, :] = [0, np.nan, float('-inf'), float('inf'), np.nan, np.nan]

        if (len(self.queue) > 0) & (
                self.server_status.loc[server_idx, 'Status'] == 1):  # assuming for now no priorities
            customer = self.queue[0]
            customer_id = customer[0]
            self.customer_type = customer[1]
            if not self.customer_type in [0, 1]:
                print('Problem')
            self.server_status.loc[server_idx, ['Status', 'Arr_num', 'Start', 'Wait', 'Type']] = \
                [2, customer_id, self.clock, self.clock - self.arrivals.loc[customer_id, 'time'], self.customer_type]
            service_time = self.generate_service()  # use type and possible other state variables
            self.server_status.loc[server_idx, 'Finish'] = self.clock + service_time
            self.summary.loc[customer_id, 'Checkin_time'] = self.clock  # complete checkin time
            self.summary.loc[customer_id, 'Exit_time'] = self.clock + service_time  # complete exit time
            self.summary.loc[customer_id, 'Service_time'] = service_time  # complete exit time
            self.summary.loc[customer_id, 'Real_WT'] = self.clock - self.summary.loc[customer_id, 'Arrival_time']
            self.list_t_abandon[customer_id] = float('inf')
            self.in_queue[self.customer_type] -= 1
            self.queue.pop(0)
        self.list_t_depart[server_idx] = self.server_status.Finish[server_idx]

    def handle_change_hour_event(self):
        '''
        When called, this method handles the modification of the servers parameter of the system.
        Updates the available and not available servers
        '''

        self.hour += 1
        if self.hour == self.max_time:
            print('Last Hour')
        print('Hour: ', self.hour)
        for indx in range(self.max_servers):
            if (self.server_status.loc[indx, 'Status'] == 1) & (self.server_schedule.iloc[indx, self.hour] == 0):
                self.server_status.iloc[indx, :] = [0, None, None, float('inf'), None, None]
            if (self.server_status.loc[indx, 'Status'] == 0) & (self.server_schedule.iloc[indx, self.hour] == 1):
                self.server_status.iloc[indx, :] = [1, None, None, float('inf'), None, None]
        self.list_t_depart = list(self.server_status['Finish'])

    def handle_abandon_event(self):
        '''
        When called, this method handles the abandonment of a customer.
        We define the Checkin time to be 'inf' and update the rest of its parameters (exit time, service time)
        We dequeue him from the queue.
        '''
        customer_id = np.argmin(self.list_t_abandon)
        self.list_t_abandon[customer_id] = float('inf')
        self.summary.loc[customer_id, ['Checkin_time', 'Service_time', 'Exit_time', 'Real_WT']] = [float('inf'), None,
                                                                                                   self.clock,
                                                                                                   self.clock -
                                                                                                   self.summary.loc[
                                                                                                       customer_id, 'Arrival_time']]
        for ind in range(len(self.queue)):
            if self.queue[ind][0] == customer_id:
                queue_idx = ind
                self.in_queue[self.queue[ind][1]] -= 1
        self.queue.pop(queue_idx)

    def generate_service(self):
        '''This method handles with the service time sample. According the init, it may depend on the speed up option or not'''
        custtype = self.customer_type
        if self.speed_up_flag:
            speed_up = math.exp(-len(self.queue) * 0.02)
        else:
            speed_up = 1
        if self.service_distribution == 'LogNormal':
            sigma = np.sqrt(np.log(2))
            mu = -0.5 * sigma * sigma
            serv1 = np.random.lognormal(mu, sigma)  # lognormal service time distribution with mean 1
            service = serv1 * self.service_mean_types[custtype][self.hour].item() * speed_up

        elif self.service_distribution == 'Exponential':
            service = np.random.exponential(self.service_mean_types[custtype][self.hour].item() * speed_up)
        return service
