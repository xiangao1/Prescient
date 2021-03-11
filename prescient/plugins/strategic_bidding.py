# Xian Gao
# Dowling Lab
# xgao1@nd.edu

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
from collections import deque, OrderedDict
from itertools import combinations

def get_data_given(df, bus=None, date=None, hour=None, generator=None, fuel_type=None):

    '''
    This function gets the data out of a pandas dataframe given one or more
    options, e.g. time.

    Arguments:
        df: the dataframe we are interested in
        bus: the bus ID we want [int]/ the bus name we want [str]
        date: the date we want [str]
        hour: the hour we want [int]
        generator: the generator ID [str]
        fuel_type: generator fuel, e.g. Coal [str]
    Returns:
        df: a dataframe that has the information we specified.
    '''

    # get data given bus id
    if bus is not None:
        # in the original rts-gmlc dataset there is a Bus ID col
        if 'Bus ID' in df.columns:
            df = df.loc[(df['Bus ID'] == bus)]

        # in the prescient result data we have to extract bus id from other col
        # e.g. gennerator name col
        elif 'Generator' in df.columns:
            # convert the type to str
            bus = str(bus)
            # find the rows that starts with the bus name
            searchrows = df['Generator'].str.startswith(bus)
            df = df.loc[searchrows,:]

        elif 'Bus' in df.columns:
            df = df.loc[(df['Bus'] == bus)]

    # get data given date
    if date is not None:
        df = df.loc[(df['Date'] == date)]

    # get data given hour
    if hour is not None:
        df = df.loc[(df['Hour'] == hour)]

    # get data given hour
    if generator is not None:

        # Similarly this is for Prescient result data
        if 'Generator' in df.columns:
            df = df.loc[(df['Generator'] == generator)]
        # this is for rts-gmlc dataset
        elif 'GEN UID' in df.columns:
            df = df.loc[(df['GEN UID'] == generator)]

    # get data given fuel
    if fuel_type is not None:
        df = df.loc[df['Fuel'] == fuel_type]

    return df

class DAM_thermal_bidding:

    n_scenario = 10

    def __init__(self,rts_gmlc_data_dir, price_forecast_dir,generators = None):

        self.price_forecast_dir = price_forecast_dir

        self.model_data = self.assemble_model_data(generator_names = generators, \
                                                   rts_gmlc_data_dir = rts_gmlc_data_dir)
        self.model = self.build_thermal_bidding_model(plan_horizon = 48,
                                                      segment_number = 4,
                                                      n_scenario = 10)

    @staticmethod
    def assemble_model_data(generator_names, rts_gmlc_data_dir, **kwargs):

        # read data
        gen_params = pd.read_csv(rts_gmlc_data_dir + 'gen.csv')
        gen_params = gen_params[gen_params['GEN UID'].isin(generator_names)]

        model_data = {}

        # generator names
        model_data['Generator'] = generator_names

        # Pmin [MW]
        model_data['Pmin'] = dict(zip(generator_names,gen_params['PMin MW']))

        # Pmax [MW]
        model_data['Pmax'] = dict(zip(generator_names,gen_params['PMax MW']))

        # minimum up time [MW/hr]
        model_data['UT'] = dict(zip(generator_names,gen_params['Min Up Time Hr'].astype(int)))

        # minimum down time [hr]
        model_data['DT'] = dict(zip(generator_names,gen_params['Min Down Time Hr'].astype(int)))

        ## ramp rates [MW/hr]
        ramp_rates = gen_params['Ramp Rate MW/Min'].values * 60

        # ramp up rate [MW/hr]
        model_data['RU'] = dict(zip(generator_names,ramp_rates))

        # ramp down rate [MW/hr]
        model_data['RD'] = dict(zip(generator_names,ramp_rates))

        # ramp start up [MW/hr]
        model_data['SU'] = {gen: min(model_data['Pmin'][gen],model_data['RU'][gen]) for gen in generator_names}

        # ramp shut down [MW/hr] (will use pmin for now)
        model_data['SD'] = {gen: min(model_data['Pmin'][gen],model_data['RD'][gen]) for gen in generator_names}

        # start up cost [$/SU] (will use the warm start up cost for now)
        start_up_cost = gen_params['Start Heat Warm MBTU'] * gen_params['Fuel Price $/MMBTU']
        model_data['SU Cost'] = dict(zip(generator_names,start_up_cost))

        ## production cost

        # power segments and marginal costs
        model_data['Power Segments'] = {}
        model_data['Marginal Costs'] = {}

        model_data['Min Load Cost'] = dict(zip(generator_names,gen_params['HR_avg_0']/1000 * gen_params['Fuel Price $/MMBTU'] * gen_params['PMin MW']))
        for gen in generator_names:
            df = get_data_given(df = gen_params, generator = gen)
            for l in range(1,4):
                # power segements
                model_data['Power Segments'][(gen,l)] = float(df['Output_pct_{}'.format(l)] * df['PMax MW'])

                # segment marginal cost
                model_data['Marginal Costs'][(gen,l)] = float(df['HR_incr_{}'.format(l)]/1000 * df['Fuel Price $/MMBTU'])

        # get the original cost curve
        model_data['Original Cost Curve'] = {}
        for gen in generator_names:
            model_data['Original Cost Curve'][gen] = OrderedDict()

            pmin = round(model_data['Pmin'][gen],2)
            model_data['Original Cost Curve'][gen][pmin] = model_data['Min Load Cost'][gen]

            old_p = pmin
            old_cost = model_data['Original Cost Curve'][gen][pmin]
            for l in range(1,4):

                new_p = round(model_data['Power Segments'][(gen,l)],2)
                delta_p = new_p - old_p

                increased_cost = delta_p * model_data['Marginal Costs'][(gen,l)]
                model_data['Original Cost Curve'][gen][new_p] = old_cost + increased_cost

                old_cost += increased_cost
                old_p = new_p

        model_data['Original Marginal Cost Curve'] = {}
        for gen in generator_names:
            model_data['Original Marginal Cost Curve'][gen] = OrderedDict()

            pmin = round(model_data['Pmin'][gen],2)
            model_data['Original Marginal Cost Curve'][gen][pmin] = model_data['Min Load Cost'][gen]/pmin

            for l in range(1,4):
                new_p = round(model_data['Power Segments'][(gen,l)],2)
                model_data['Original Marginal Cost Curve'][gen][new_p] = model_data['Marginal Costs'][(gen,l)]

        for key in kwargs:
            model_data[key] = {gen: kwargs[key] for gen in generator_names}

        return model_data

    @staticmethod
    def _add_UT_DT_constraints(m):

        def pre_shut_down_trajectory_set_rule(m):
            return ((j,t) for j in m.UNITS for t in range(-pyo.value(m.min_dw_time[j]) + 1,0))
        m.pre_shut_down_trajectory_set = pyo.Set(dimen = 2,initialize = pre_shut_down_trajectory_set_rule, ordered = True)

        def pre_start_up_trajectory_set_rule(m):
            return ((j,t) for j in m.UNITS for t in range(-pyo.value(m.min_up_time[j]) + 1,0))
        m.pre_start_up_trajectory_set = pyo.Set(dimen = 2,initialize = pre_start_up_trajectory_set_rule, ordered = True)

        m.pre_shut_down_trajectory = pyo.Param(m.pre_shut_down_trajectory_set, initialize = 0, mutable = True)
        m.pre_start_up_trajectory = pyo.Param(m.pre_start_up_trajectory_set, initialize = 0, mutable = True)

        def min_down_time_rule(m,j,h,k):
            if h < pyo.value(m.min_dw_time[j]):
                return sum(m.pre_shut_down_trajectory[j,h0] for h0 in range(h - pyo.value(m.min_dw_time[j]) + 1,0)) \
                       + sum(m.shut_dw[j,h0,k] for h0 in range(h + 1)) <= 1 - m.on_off[j,h,k]
            else:
                return sum(m.shut_dw[j,h0,k] for h0 in range(h - pyo.value(m.min_dw_time[j]) + 1, h + 1)) <= 1 - m.on_off[j,h,k]
        m.min_down_time_con = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = min_down_time_rule)

        def min_up_time_rule(m,j,h,k):
            if h < pyo.value(m.min_up_time[j]):
                return sum(m.pre_start_up_trajectory[j,h0] for h0 in range(h - pyo.value(m.min_up_time[j]) + 1,0)) \
                       + sum(m.start_up[j,h0,k] for h0 in range(h + 1)) <= m.on_off[j,h,k]
            else:
                return sum(m.start_up[j,h0,k] for h0 in range(h - pyo.value(m.min_up_time[j]) + 1, h + 1)) <= m.on_off[j,h,k]
        m.min_up_time_con = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = min_up_time_rule)

        return

    def build_thermal_bidding_model(self,
                                    plan_horizon = 48,
                                    segment_number = 4,
                                    n_scenario = 10):

        model_data = self.model_data
        m = pyo.ConcreteModel()

        ## define the sets
        m.HOUR = pyo.Set(initialize = range(plan_horizon))
        m.SEGMENTS = pyo.Set(initialize = range(1, segment_number))
        m.UNITS = pyo.Set(initialize = model_data['Generator'], ordered = True)
        m.SCENARIOS = pyo.Set(initialize = range(n_scenario))

        ## define the parameters

        # add power schedule as a Param
        m.power_dispatch = pyo.Param(m.UNITS,m.HOUR, initialize = 0, mutable = True)

        m.DAM_price = pyo.Param(m.HOUR,m.SCENARIOS,initialize = 20, mutable = True)

        m.start_up_cost = pyo.Param(m.UNITS,initialize = model_data['SU Cost'],mutable = False)

        # capacity of generators: upper bound (MW)
        m.Pmax = pyo.Param(m.UNITS,initialize = model_data['Pmax'], mutable = False)

        # minimum power of generators: lower bound (MW)
        m.Pmin = pyo.Param(m.UNITS,initialize = model_data['Pmin'], mutable = False)

        m.power_segment_bounds = pyo.Param(m.UNITS,m.SEGMENTS,initialize = model_data['Power Segments'], mutable = False)

        # get the cost slopes
        m.F = pyo.Param(m.UNITS,m.SEGMENTS,initialize = model_data['Marginal Costs'], mutable = False)

        m.min_load_cost = pyo.Param(m.UNITS,initialize = model_data['Min Load Cost'], mutable = False)

        # Ramp up limits (MW/h)
        m.ramp_up = pyo.Param(m.UNITS,initialize = model_data['RU'], mutable = False)

        # Ramp down limits (MW/h)
        m.ramp_dw = pyo.Param(m.UNITS,initialize = model_data['RD'], mutable = False)

        # start up ramp limit
        m.ramp_start_up = pyo.Param(m.UNITS,initialize = model_data['SU'], mutable = False)

        # shut down ramp limit
        m.ramp_shut_dw = pyo.Param(m.UNITS,initialize = model_data['SD'], mutable = False)

        # minimum down time [hr]
        m.min_dw_time = pyo.Param(m.UNITS,initialize = model_data['DT'], mutable = False)

        # minimum up time [hr]
        m.min_up_time = pyo.Param(m.UNITS,initialize = model_data['UT'], mutable = False)

        # power from the previous day (MW)
        # need to assume the power output is at least always at the minimum pow output

        # on/off status from previous day
        m.pre_on_off = pyo.Param(m.UNITS,within = pyo.Binary,default= 1,mutable = True)

        # define a function to initialize the previous power params
        def init_pre_pow_fun(m,j):
            return m.pre_on_off[j]*m.Pmin[j]
        m.pre_P_T = pyo.Param(m.UNITS,initialize = init_pre_pow_fun, mutable = True)

        ## define the variables

        # generator power (MW)

        # power generated by thermal generator
        m.P_T = pyo.Var(m.UNITS,m.HOUR,m.SCENARIOS,within = pyo.NonNegativeReals)

        # binary variables indicating on/off
        m.on_off = pyo.Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = True, within = pyo.Binary)

        # binary variables indicating  start_up
        m.start_up = pyo.Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = False, within = pyo.Binary)

        # binary variables indicating shut down
        m.shut_dw = pyo.Var(m.UNITS,m.HOUR,m.SCENARIOS,initialize = False, within = pyo.Binary)

        # power produced in each segment
        m.power_segment = pyo.Var(m.UNITS,m.HOUR,m.SEGMENTS, m.SCENARIOS, within = pyo.NonNegativeReals)

        ## Constraints

        # bounds on gen_pow
        def lhs_bnd_gen_pow_fun(m,j,h,k):
            return m.on_off[j,h,k] * m.Pmin[j] <= m.P_T[j,h,k]
        m.lhs_bnd_gen_pow = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = lhs_bnd_gen_pow_fun)

        def rhs_bnd_gen_pow_fun(m,j,h,k):
            return m.P_T[j,h,k] <= m.on_off[j,h,k] * m.Pmax[j]
        m.rhs_bnd_gen_pow = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = rhs_bnd_gen_pow_fun)

        # linearized power
        def linear_power_fun(m,j,h,k):
            return m.P_T[j,h,k] == \
            sum(m.power_segment[j,h,l,k] for l in m.SEGMENTS) + m.Pmin[j]*m.on_off[j,h,k]
        m.linear_power = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = linear_power_fun)

        # bounds on segment power
        def seg_pow_bnd_fun(m,j,h,l,k):
            if l == 1:
                return m.power_segment[j,h,l,k]<= (m.power_segment_bounds[j,l] - m.Pmin[j]) * m.on_off[j,h,k]
            else:
                return m.power_segment[j,h,l,k]<= (m.power_segment_bounds[j,l] - m.power_segment_bounds[j,l-1]) * m.on_off[j,h,k]
        m.seg_pow_bnd = pyo.Constraint(m.UNITS,m.HOUR,m.SEGMENTS,m.SCENARIOS,rule = seg_pow_bnd_fun)

        # start up and shut down logic (Arroyo and Conejo 2000)
        def start_up_shut_dw_fun(m,j,h,k):
            if h == 0:
                return m.start_up[j,h,k] - m.shut_dw[j,h,k] == m.on_off[j,h,k] - m.pre_on_off[j]
            else:
                return m.start_up[j,h,k] - m.shut_dw[j,h,k] == m.on_off[j,h,k] - m.on_off[j,h-1,k]
        m.start_up_shut_dw = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = start_up_shut_dw_fun)

        # either start up or shut down
        def start_up_or_shut_dw_fun(m,j,h,k):
            return m.start_up[j,h,k] + m.shut_dw[j,h,k] <= 1
        m.start_up_or_shut_dw = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = start_up_or_shut_dw_fun)

        # ramp up limits
        def ramp_up_fun(m,j,h,k):
            '''
            j,h stand for unit, hour,scenario respectively.
            '''
            if h==0:
                return m.P_T[j,h,k] <= m.pre_P_T[j] \
                + m.ramp_up[j]*m.pre_on_off[j]\
                + m.ramp_start_up[j]*m.start_up[j,h,k]
            else:
                return m.P_T[j,h,k] <= m.P_T[j,h-1,k] \
                + m.ramp_up[j]*m.on_off[j,h-1,k]\
                + m.ramp_start_up[j]*m.start_up[j,h,k]
        m.ramp_up_con = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_up_fun)

        # ramp shut down limits
        def ramp_shut_dw_fun(m,j,h,k):
            '''
            j,h stand for unit, hour,scenario respectively.
            '''
            if h==0:
                return m.pre_P_T[j] <= m.Pmax[j]*m.on_off[j,h,k] + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
            else:
                return m.P_T[j,h-1,k] <= m.Pmax[j]*m.on_off[j,h,k] + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
        m.ramp_shut_dw_con = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_shut_dw_fun)

        # ramp down limits
        def ramp_dw_fun(m,j,h,k):
            '''
            j,h stand for unit, hour,scenario respectively.
            '''
            if h == 0:
                return m.pre_P_T[j] - m.P_T[j,h,k] <= m.ramp_dw[j] * m.on_off[j,h,k]\
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
            else:
                return m.P_T[j,h-1,k] - m.P_T[j,h,k] <= m.ramp_dw[j] * m.on_off[j,h,k]\
                + m.ramp_shut_dw[j] * m.shut_dw[j,h,k]
        m.ramp_dw_con = pyo.Constraint(m.UNITS,m.HOUR,m.SCENARIOS,rule = ramp_dw_fun)

        ## add min up and down time constraints
        self._add_UT_DT_constraints(m)

        ## add bidding constraints
        self._add_bidding_constraints(m)

        ## Expression

        def prod_cost_fun(m,j,h,k):
            return m.min_load_cost[j] * m.on_off[j,h,k] \
            + sum(m.F[j,l]*m.power_segment[j,h,l,k] for l in m.SEGMENTS)
        m.prod_cost_approx = pyo.Expression(m.UNITS,m.HOUR,m.SCENARIOS,rule = prod_cost_fun)

        # start up costs
        def start_cost_fun(m,j,h,k):
            return m.start_up_cost[j]*m.start_up[j,h,k]
        m.start_up_cost_expr = pyo.Expression(m.UNITS,m.HOUR,m.SCENARIOS,rule = start_cost_fun)

        # total cost
        def tot_cost_fun(m,j,h,k):
            return m.prod_cost_approx[j,h,k] + m.start_up_cost_expr[j,h,k]
        m.tot_cost = pyo.Expression(m.UNITS,m.HOUR,m.SCENARIOS,rule = tot_cost_fun)

        ## Objective
        def exp_revenue_fun(m):
            return sum(m.P_T[j,h,k]*m.DAM_price[h,k]- m.tot_cost[j,h,k] for h in m.HOUR for j in m.UNITS for k in m.SCENARIOS)
        m.exp_revenue = pyo.Objective(rule = exp_revenue_fun,sense = pyo.maximize)

        return m

    @staticmethod
    def _update_UT_DT(m,implemented_shut_down, implemented_start_up):
        '''
        implemented_shut_down: {unit: []}
        implemented_start_up: {unit: []}
        '''

        # copy to a queue
        pre_shut_down_trajectory_copy = {}
        pre_start_up_trajectory_copy = {}

        for unit in m.UNITS:
            pre_shut_down_trajectory_copy[unit] = deque([])
            pre_start_up_trajectory_copy[unit] = deque([])

        for unit,t in m.pre_shut_down_trajectory_set:
            pre_shut_down_trajectory_copy[unit].append(round(pyo.value(m.pre_shut_down_trajectory[unit,t])))
        for unit,t in m.pre_start_up_trajectory_set:
            pre_start_up_trajectory_copy[unit].append(round(pyo.value(m.pre_start_up_trajectory[unit,t])))

        # add implemented trajectory to the queue
        for unit in m.UNITS:
            pre_shut_down_trajectory_copy[unit] += deque(implemented_shut_down[unit])
            pre_start_up_trajectory_copy[unit] += deque(implemented_start_up[unit])

        # pop out outdated trajectory
        for unit in m.UNITS:

            while len(pre_shut_down_trajectory_copy[unit]) > pyo.value(m.min_dw_time[unit]) - 1:
                pre_shut_down_trajectory_copy[unit].popleft()
            while len(pre_start_up_trajectory_copy[unit]) > pyo.value(m.min_up_time[unit]) - 1:
                pre_start_up_trajectory_copy[unit].popleft()

        # actual update
        for unit,t in m.pre_shut_down_trajectory_set:
            m.pre_shut_down_trajectory[unit,t] = pre_shut_down_trajectory_copy[unit].popleft()

        for unit,t in m.pre_start_up_trajectory_set:
            m.pre_start_up_trajectory[unit,t] = pre_start_up_trajectory_copy[unit].popleft()

        return

    @staticmethod
    def _update_power(m,implemented_power_output):
        '''
        implemented_power_output: {unit: []}
        '''

        for unit in m.UNITS:
            m.pre_P_T[unit] = round(implemented_power_output[unit][-1],2)
            m.pre_on_off[unit] = round(int(implemented_power_output[unit][-1] > 1e-3))

        return

    def update_model(self,implemented_power_output,implemented_shut_down, implemented_start_up):

        self._update_UT_DT(self.model,implemented_shut_down, implemented_start_up)
        self._update_power(self.model,implemented_power_output)

        return

    # define a function to add bidding constraints to models
    @staticmethod
    def _add_bidding_constraints(m,NA_con_range = 24):

        '''
        This function takes the thermal model and add bidding cosntraints
        to it in order to bid into DAM.

        Note: it is a little different in the paper from what I have done with the
        energy storage system model. This constraint is enforced based on the energy
        output instead of power output.
        '''

        # generate scenarios combinations
        scenario_comb = combinations(m.SCENARIOS,2)

        # constraints for thermal generators
        def bidding_con_fun1(m):
            for k in scenario_comb:
                for j in m.UNITS:
                    for h in range(NA_con_range):
                        yield (m.P_T[j,h,k[0]] - m.P_T[j,h,k[1]])\
                        *(m.DAM_price[h,k[0]] - m.DAM_price[h,k[1]]) >= 0
        m.bidding_con1 = pyo.ConstraintList(rule = bidding_con_fun1)

        return

    # define a function to add non_anticipativity constraints to models
    @staticmethod
    def add_non_anticipativity(m,NA_con_range = 2):

        '''
        This function takes the thermal model and add non non_anticipativity cosntraints
        to it in order to do self-scheduling in DAM.

        '''
        from itertools import combinations

        # generate scenarios combinations
        scenario_comb = combinations(m.SCENARIOS,2)

        def NAC_list_rules(m):
            for k in scenario_comb:
                for j in m.UNITS:
                    for h in range(NA_con_range):
                        yield m.gen_pow[j,h,k[0]] == m.gen_pow[j,h,k[1]]
        m.NAC_SS_cons = ConstraintList(rule = NAC_list_rules)

        return m

    @staticmethod
    def extract_power(m):

        '''
        This function takes the solved thermal model to extract power output (MW) in
        each hour.

        Input:
        1. a pyomo model

        Output:
        1.  a dictionary whose keys are units and the values are numpy
        arrays (n_scenario*horizon) which store the energy output at each time in each scenario.
        '''

        power_output = {}

        for j in m.UNITS:
            unit_output = np.zeros((len(m.SCENARIOS),len(m.HOUR)))
            for k in m.SCENARIOS:
                for h in m.HOUR:
                    unit_output[k,h]=round(value(m.gen_pow[j,h,k]),1)

            power_output[j] = unit_output

        return power_output

    # define a fucntion to get the bidding power and corresponding prices
    def get_bid_power_price_old(self,m_bid,price_forecast,plan_horizon = 48,verbose = False):

        '''
        Get the bidding curves of the generator from the result of SP.
        '''

        assert plan_horizon > 24

        power_output = self.extract_power(m_bid)

        # sort the price forecast along scenario axis
        price_forecast_sorted = np.sort(price_forecast,axis = 0)

        power_output_dict = {}
        marginal_cost_dict = {}
        cost_dict = {}

        for unit in m_bid.UNITS:

            # sort the energy output in this unit, along scenario axis
            power_unit_sorted = np.sort(power_output[unit],axis = 0)

            # initialize lists to store the result
            power_unit_list = []
            price_unit_list = []
            cost_unit_list = []

            for h in range(plan_horizon):

                if h < 24:

                    # find power less than pmin and delete
                    del_inx = np.where(power_unit_sorted[:,h]<self.min_pow[unit])[0]
                    power_unit = np.delete(power_unit_sorted[:,h],del_inx)
                    price_forecast_unit = np.delete(price_forecast_sorted[:,h],del_inx)

                    # if the unit bids to be turned off, we bid into the true costs
                    if len(power_unit) == 0:

                        power_unit = self.output[unit][1:]
                        price_forecast_unit = self.prod_cost[unit]

                    # make sure original output points in the list
                    for p_idx, p in enumerate(self.output[unit]):

                        if p_idx == 0:
                            continue

                        if p in power_unit:
                            idx = np.where(power_unit==p)[0][0]
                            price_forecast_unit[idx] = max(price_forecast_unit[idx],self.prod_cost[unit][p_idx-1])
                        else:
                            # add p to corresponding position
                            insert_idx = np.searchsorted(power_unit,p)
                            power_unit = np.insert(power_unit,insert_idx,p)
                            price_forecast_unit = np.insert(price_forecast_unit,insert_idx,\
                            self.prod_cost[unit][p_idx-1])

                    # make sure pmin and pmax are in the curve
                    # if float(self.min_pow[unit]) in power_unit:
                    #     pass
                    # else:
                    #     # add pmin to the power output array
                    #     power_unit = np.insert(power_unit,0,self.min_pow[unit])
                    #     price_forecast_unit = np.insert(price_forecast_unit,0,\
                    #     self.prod_cost[unit][0])
                    #
                    # if float(self.capacity[unit]) in power_unit:
                    #     pass
                    # else:
                    #     # add pmax to power output array
                    #     power_unit = np.append(power_unit,self.capacity[unit])
                    #     price_forecast_unit = np.append(price_forecast_unit,\
                    #     self.prod_cost[unit][-1])

                    # calculate the unique elements in the power output and prices
                    power_unit,unique_power_idx = np.unique(power_unit,\
                    return_index=True)

                    price_forecast_unit = price_forecast_unit[unique_power_idx]
                    cost_unit = np.cumsum(np.diff(np.insert(power_unit,0,0))\
                    * price_forecast_unit)

                    # record the offering power and prices
                    power_unit_list.append(power_unit)
                    price_unit_list.append(price_forecast_unit)
                    cost_unit_list.append(cost_unit)

                    if verbose:
                        print("Unit {} Power {}\nMarginal Costs {}".format(unit,\
                        power_unit,price_forecast_unit))

                else:

                    power_unit = self.output[unit][1:]
                    price_forecast_unit = self.prod_cost[unit]
                    cost_unit = np.cumsum(np.diff(np.insert(power_unit,0,0))\
                    * price_forecast_unit)

                    # record the offering power and prices
                    power_unit_list.append(power_unit)
                    price_unit_list.append(price_forecast_unit)
                    cost_unit_list.append(cost_unit)

            power_output_dict[unit] = power_unit_list
            marginal_cost_dict[unit] = price_unit_list
            cost_dict[unit] = cost_unit_list

        return power_output_dict, marginal_cost_dict, cost_dict

    def stochastic_bidding(self,date):
        '''
        Solve the stochastic bidding optimization problem.
        '''

        print("")
        print("In stochastic_bidding\n")

        m = self.model

        # read forecasts
        price_forecast_file = os.path.join(self.price_forecast_dir,\
                                           'date={}_lmp_forecasts.csv'.format(date))
        forecasts = pd.read_csv(price_forecast_file,header = None).values

        # pass the forecast into pyomo model
        for i in m.SCENARIOS:
            for t in m.HOUR:
                m.DAM_price[t,i] = forecasts[i,t]

        # solve the model
        solver = pyo.SolverFactory('gurobi')
        result = solver.solve(m,tee=True)

        # extract the bids out from the model
        bids = self.get_bid_power_price()

        return bids

    def get_bid_power_price(self):

        '''
        return bids {t: {gen:{power: cost}}}
        '''

        bids = {}

        for t in self.model.HOUR:
            bids[t] = {}
            for gen in self.model.UNITS:

                # get the pmin of the gen
                pmin = self.model_data['Pmin'][gen]

                # declare a dict to store the pmin
                temp_bids = {}

                # add the bids from the model
                for k in self.model.SCENARIOS:
                    power = round(pyo.value(self.model.P_T[gen,t,k]),2)
                    marginal_cost = pyo.value(self.model.DAM_price[t,k])

                    if power < pmin:
                        continue

                    # add bids without duplicates
                    if power in temp_bids:
                        temp_bids[power] = min(temp_bids[power],marginal_cost)
                    else:
                        temp_bids[power] = marginal_cost

                # make sure the orignal points in the bids
                for power, marginal_cost in self.model_data['Original Marginal Cost Curve'][gen].items():

                    if power in temp_bids:
                        pass
                    else:
                        temp_bids[power] = marginal_cost

                # sort the curves by power
                temp_bids = OrderedDict(sorted(temp_bids.items()))

                # make sure the curve is nondecreasing
                pre_power = pmin
                for power, marginal_cost in temp_bids.items():

                    # ignore pmin, because min load cost is special
                    if pre_power == pmin:
                        pre_power = power
                        continue
                    temp_bids[power] = max(temp_bids[power],temp_bids[pre_power])

                # calculate the actual cost
                pre_power = 0
                pre_cost = 0
                for power, marginal_cost in temp_bids.items():

                    delta_p = power - pre_power
                    temp_bids[power] = pre_cost + marginal_cost * delta_p
                    pre_power = power
                    pre_cost += marginal_cost * delta_p

                bids[t][gen] = temp_bids

        return bids

    def record_bids(self,ruc_date):
        '''
        cols: gen, date, hour, power 1, ..., power n, price 1, ..., price n
        '''
        pass
