from optparse import Option
import prescient.plugins

# Add command line options
opt = Option('--track-ruc-signal',
             help='When tracking the market signal, RUC signals are used instead of the SCED signal.',
             action='store_true',
             dest='track_ruc_signal',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--track-sced-signal',
             help='When tracking the market signal, SCED signals are used instead of the RUC signal.',
             action='store_true',
             dest='track_sced_signal',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--hybrid-tracking',
             help='When tracking the market signal, hybrid model is used.',
             action='store_true',
             dest='hybrid_tracking',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--track-horizon',
             help="Specifies the number of hours in the look-ahead horizon "
                  "when each tracking process is executed.",
             action='store',
             dest='track_horizon',
             type='int',
             default=48)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--bidding-generator',
             help="Specifies the generator we derive bidding strategis for.",
             action='store',
             dest='bidding_generator',
             type='string',
             default='102_STEAM_3')
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--bidding',
             help="Invoke generator strategic bidding when simulate.",
             action='store_true',
             dest='bidding',
             default=False)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--deviation-weight',
             help="Set the weight for deviation term when tracking",
             action='store',
             dest='deviation_weight',
             type='float',
             default=30)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--ramping-weight',
             help="Set the weight for ramping term when tracking",
             action='store',
             dest='ramping_weight',
             type='float',
             default=20)
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--cost-weight',
             help="Set the weight for cost term when tracking",
             action='store',
             dest='cost_weight',
             type='float',
             default=1)
prescient.plugins.add_custom_commandline_option(opt)

### End add new command line options ###

from strategic_bidding import DAM_thermal_bidding
from tracking import DAM_thermal_tracking
import pyomo.environ as pyo
import dateutil.parser
import numpy as np
from pandas import read_csv
import os

def initialize_customized_results(options, simulator):

    simulator.data_manager.extensions['customized_results'] = {}
    customized_results = simulator.data_manager.extensions['customized_results']

    customized_results['Generator'] = []
    customized_results['Date'] = []
    customized_results['Hour'] = []
    customized_results['State'] = []
    customized_results['RUC Schedule'] = []
    customized_results['SCED Schedule'] = []
    customized_results['Power Output'] = []

    return
prescient.plugins.register_initialization_callback(initialize_customized_results)

def initialize_bidding_object(options, simulator):

    # initialize the model class
    thermal_bid = DAM_thermal_bidding(n_scenario=10)
    simulator.data_manager.extensions['thermal_bid'] = thermal_bid

    return
prescient.plugins.register_initialization_callback(initialize_bidding_object)

def initialize_tracking_object(options, simulator):

    # initialize the model class
    thermal_track = DAM_thermal_tracking(n_scenario=10)
    simulator.data_manager.extensions['thermal_track'] = thermal_track

    return
prescient.plugins.register_initialization_callback(initialize_tracking_object)

def bid_into_DAM(options, simulator, ruc_instance, ruc_date, ruc_hour):

    ## TODO: make sure the model is updated


    thermal_bid = simulator.data_manager.extensions['thermal_bid']

    # generate bids
    thermal_bid.stochastic_bidding(thermal_bid.model,ruc_date)

    # pass to prescient
    thermal_bid.pass_bid_to_prescient(options, simulator, ruc_instance, ruc_date, ruc_hour)

    # record bids
    thermal_bid.record_bids()

    return
prescient.plugins.register_before_ruc_solve_callback(bid_into_DAM)

def save_ruc_plan(options, simulator, ruc_plan, ruc_date, ruc_hour):

    # save the ruc plan as a property of tracking object

    pass
prescient.plugins.register_after_ruc_generation_callback(save_ruc_plan)

def track_sced_signal(options, simulator, sced_instance):

    ## TODO: actual tracking

    ## TODO: record operation results

    ## update the tracking model
    pass
prescient.plugins.register_after_operations_callback(track_sced_signal)

def update_observed_thermal_dispatch(options, simulator, ops_stats):

    current_time = simulator.time_manager.current_time
    h = current_time.hour
    date_as_string = current_time.date
    date_idx = simulator.time_manager.dates_to_simulate.index(date_as_string)

    total_power_delivered_arr = simulator.data_manager.extensions['total_power_delivered_arr']

    if options.track_ruc_signal:
        print('Making changes in observed power output using tracking RUC model.')
        g = options.bidding_generator
        ops_stats.observed_thermal_dispatch_levels[g] = total_power_delivered_arr[h,date_idx]

    elif options.track_sced_signal:
        print('Making changes in observed power output using tracking SCED model.')
        g = options.bidding_generator
        ops_stats.observed_thermal_dispatch_levels[g] = total_power_delivered_arr[h,date_idx]

prescient.plugins.register_update_operations_stats_callback(update_observed_thermal_dispatch)


def after_ruc_activation(options, simulator):

    # change the ruc plan in tracking object
    pass
prescient.plugins.register_after_ruc_activation_callback(after_ruc_activation)

def write_customize_results(options, simulator):

    '''
    write results in bidding and tracking objects
    '''
    pass
prescient.plugins.register_after_simulation_callback(write_customize_results)
'''
def get_gpoints_gvalues(cost_curve_store_dir,date,gen_name = '102_STEAM_3',horizon = 24,verbose = False):

    gpoints = {}
    gvalues = {}

    # read the csv file
    for h in range(horizon):

        if verbose:
            print("")
            print("Getting cost curve from Date: {}, Hour: {}.".format(date,h))

        gpoints[h] = list(read_csv(cost_curve_store_dir+gen_name+\
        '_date={}_hour={}_cost_curve.csv'.format(date,h),header = None).values[:,0])
        gvalues[h] = list(read_csv(cost_curve_store_dir+gen_name+\
        '_date={}_hour={}_cost_curve.csv'.format(date,h),header = None).values[:,1])

    return gpoints,gvalues

def initialize_plugin(options, simulator):
    # Xian: add 2 np arrays to store RUC and SCED schedules for the interested generator
    simulator.data_manager.extensions['ruc_schedule_arr'] = np.zeros((24,options.num_days))
    simulator.data_manager.extensions['sced_schedule_arr'] = np.zeros((24,options.num_days))

    # Xian: add 2 np arrays to store
    # 1. the actual total power output from the hybrid system
    # 2. power output from the thermal generator in the hybrid system
    simulator.data_manager.extensions['total_power_delivered_arr'] = np.zeros((24,options.num_days)) # P_R
    simulator.data_manager.extensions['thermal_power_delivered_arr'] = np.zeros((24,options.num_days)) # P_G
    simulator.data_manager.extensions['thermal_power_generated_arr'] = np.zeros((24,options.num_days)) # P_T

    # initialize the model class
    thermal_bid = DAM_thermal_bidding(n_scenario=10)
    simulator.data_manager.extensions['thermal_bid'] = thermal_bid

    first_date = str(dateutil.parser.parse(options.start_date).date())

    # build bidding model
    if options.bidding:
        m_bid = thermal_bid.create_bidding_model(generator = options.bidding_generator)
        price_forecast_dir = '../../prescient/plugins/price_forecasts/'
        cost_curve_store_dir = '../../prescient/plugins/cost_curves/'

        simulator.data_manager.extensions['cost_curve_store_dir'] = cost_curve_store_dir
        simulator.data_manager.extensions['price_forecast_dir'] = price_forecast_dir
        simulator.data_manager.extensions['m_bid'] = m_bid

    # build tracking model
    if options.track_ruc_signal:
        print('Building a track model for RUC signals.')
        simulator.data_manager.extensions['m_track_ruc'] = thermal_bid.build_tracking_model(options.ruc_horizon, \
                                                                                            generator = options.bidding_generator,\
                                                                                            track_type = 'RUC', \
                                                                                            hybrid = options.hybrid_tracking)

    elif options.track_sced_signal:
        print('Building a track model for SCED signals.')
        m_track_sced = thermal_bid.build_tracking_model(options.sced_horizon,\
                                                        generator = options.bidding_generator,\
                                                        track_type = 'SCED',\
                                                        hybrid = options.hybrid_tracking)
        simulator.data_manager.extensions['m_track_sced'] = m_track_sced


        # initialize a list/array to record the power output of sced tracking
        # model in real-time, so it can update the initial condition properly
        # every hour
        simulator.data_manager.extensions['sced_tracker_power_record'] = \
            {options.bidding_generator:
             np.repeat(pyo.value(m_track_sced.pre_pow[options.bidding_generator]),\
                       pyo.value(m_track_sced.pre_up_hour[options.bidding_generator]))
            }

prescient.plugins.register_initialization_callback(initialize_plugin)

def tweak_ruc_before_solve(options, simulator, ruc_instance, ruc_date, ruc_hour):

    if not options.bidding:
        return

    print("Getting cost cuves for UC.\n")
    current_time = simulator.time_manager.current_time

    thermalBid = simulator.data_manager.extensions['thermal_bid']
    m_bid = simulator.data_manager.extensions['m_bid']

    price_forecast_dir = simulator.data_manager.extensions['price_forecast_dir']
    cost_curve_store_dir = simulator.data_manager.extensions['cost_curve_store_dir']

    if current_time is not None:
        date_as_string = ruc_date

        current_date = current_time.date
        date_idx = simulator.time_manager.dates_to_simulate.index(current_date)

        # Xian: solve bidding problem here
        ruc_schedule_arr = simulator.data_manager.extensions['ruc_schedule_arr'][:,date_idx]
        thermalBid.update_model_params(m_bid,\
                                       ruc_schedule_arr,\
                                       unit = options.bidding_generator)
        thermalBid.reset_constraints(m_bid,options.ruc_horizon)

    else: # first RUC solve
        date_as_string = ruc_date

    # solve the bidding model for the first simulation day
    thermalBid.stochastic_bidding(m_bid,price_forecast_dir,cost_curve_store_dir,date_as_string)

    gen_name = options.bidding_generator
    gpoints, gvalues = get_gpoints_gvalues(cost_curve_store_dir,\
                                            date=date_as_string,\
                                            gen_name=gen_name,\
                                            horizon = options.ruc_horizon)
    gen_dict = ruc_instance.data['elements']['generator'][gen_name]

    p_cost = [[(gpnt, gval) for gpnt, gval in zip(gpoints[t], gvalues[t])] for t in range(options.ruc_horizon)]

    gen_dict['p_cost'] = {'data_type': 'time_series',
                          'values': [{'data_type' : 'cost_curve',
                                     'cost_curve_type':'piecewise',
                                     'values':p_cost[t]} for t in range(options.ruc_horizon)]
                         }

prescient.plugins.register_before_ruc_solve_callback(tweak_ruc_before_solve)

def after_ruc(options, simulator, ruc_plan, ruc_date, ruc_hour):

    ruc_instance = ruc_plan.deterministic_ruc_instance

    date_idx = simulator.time_manager.dates_to_simulate.index(ruc_date)

    gen_name = options.bidding_generator

    g_dict = ruc_instance.data['elements']['generator'][gen_name]
    ruc_dispatch_level_for_next_period = {gen_name: g_dict['pg']['values']}

    is_first_date = (date_idx==0)
    if is_first_date:
        simulator.data_manager.extensions['ruc_dispatch_level_current'] = \
                ruc_dispatch_level_for_next_period
    else:
        simulator.data_manager.extensions['ruc_dispatch_level_for_next_period'] = \
                ruc_dispatch_level_for_next_period

    ruc_schedule_arr = simulator.data_manager.extensions['ruc_schedule_arr']

    # record the ruc signal
    ruc_schedule_arr[:,date_idx] = np.array(ruc_dispatch_level_for_next_period[options.bidding_generator]).flatten()[:24]

    if options.track_ruc_signal:

        m_track_ruc = simulator.data_manager.extensions['m_track_ruc']
        thermalBid = simulator.data_manager.extensions['thermal_bid']

        thermalBid.pass_schedule_to_track_and_solve(m_track_ruc,\
                                                    ruc_dispatch_level_for_next_period,\
                                                    RT_price=None,\
                                                    deviation_weight = options.deviation_weight, \
                                                    ramping_weight = options.ramping_weight,\
                                                    cost_weight = options.cost_weight)


        # record the track power output profile
        if options.hybrid_tracking == False:
            track_gen_pow_ruc = thermalBid.extract_pow_s_s(m_track_ruc, \
                                                           horizon = 24, \
                                                           verbose = False)
            thermal_track_gen_pow_ruc = track_gen_pow_ruc
            thermal_generated_ruc = track_gen_pow_ruc
        else:
            track_gen_pow_ruc,\
            thermal_track_gen_pow_ruc,\
            thermal_generated_ruc = thermalBid.extract_pow_s_s(m_track_ruc, \
                                                               horizon = 24, \
                                                               hybrid = True, \
                                                               verbose = False)

        # record the total power delivered and thermal power delivered
        total_power_delivered_arr = simulator.data_manager.extensions['total_power_delivered_arr']
        thermal_power_delivered_arr = simulator.data_manager.extensions['thermal_power_delivered_arr']
        thermal_power_generated_arr = simulator.data_manager.extensions['thermal_power_generated_arr']

        total_power_delivered_arr[:,date_idx] = track_gen_pow_ruc[options.bidding_generator]
        thermal_power_delivered_arr[:,date_idx] = thermal_track_gen_pow_ruc[options.bidding_generator]
        thermal_power_generated_arr[:,date_idx] = thermal_generated_ruc[options.bidding_generator]

        # update the track model
        thermalBid.update_model_params(m_track_ruc,\
                                       thermal_generated_ruc[options.bidding_generator],\
                                       unit = options.bidding_generator,\
                                       hybrid = options.hybrid_tracking)
        thermalBid.reset_constraints(m_track_ruc,options.ruc_horizon)

prescient.plugins.register_after_ruc_generation_callback(after_ruc)

def tweak_sced_before_solve(options, simulator, sced_instance):
    current_time = simulator.time_manager.current_time
    hour = current_time.hour
    date_as_string = current_time.date

    gen_name = options.bidding_generator
    gpoints, gvalues = get_gpoints_gvalues(simulator.data_manager.extensions['cost_curve_store_dir'],
                                            date=date_as_string, gen_name=gen_name)
    gen_dict = sced_instance.data['elements']['generator'][gen_name]

    p_cost = [(gpnt, gval) for gpnt, gval in zip(gpoints[hour], gvalues[hour])]

    gen_dict['p_cost'] = {'data_type' : 'cost_curve',
                          'cost_curve_type':'piecewise',
                          'values':p_cost
                         }

prescient.plugins.register_before_operations_solve_callback(tweak_sced_before_solve)

def after_sced(options, simulator, sced_instance):

    current_time = simulator.time_manager.current_time
    h = current_time.hour
    date_as_string = current_time.date

    date_idx = simulator.time_manager.dates_to_simulate.index(date_as_string)

    gen_name = options.bidding_generator
    g_dict = sced_instance.data['elements']['generator'][gen_name]
    sced_dispatch_level = {gen_name: g_dict['pg']['values']}

    sced_schedule_arr = simulator.data_manager.extensions['sced_schedule_arr']
    sced_schedule_arr[h,date_idx] = sced_dispatch_level[options.bidding_generator][0]

    ## TODO: pass the real-time price into the function here
    # get lmps in the current planning horizon
    # get_lmps_for_deterministic_sced(lmp_sced_instance, max_bus_label_length=max_bus_label_length)
    # currently we don't need lmp, because the objective is to minimize the cost

    if options.track_sced_signal:
        ruc_dispatch_level_current = simulator.data_manager.extensions['ruc_dispatch_level_current']
        # slice the ruc dispatch for function calling below
        ruc_dispatch_level_for_current_sced_track = {options.bidding_generator:\
                        ruc_dispatch_level_current[gen_name][h:h+options.sced_horizon]}

        thermalBid = simulator.data_manager.extensions['thermal_bid']
        m_track_sced = simulator.data_manager.extensions['m_track_sced']
        thermalBid.pass_schedule_to_track_and_solve(m_track_sced,\
                                                    ruc_dispatch_level_for_current_sced_track,\
                                                    SCED_dispatch = sced_dispatch_level,\
                                                    deviation_weight = options.deviation_weight, \
                                                    ramping_weight = options.ramping_weight,\
                                                    cost_weight = options.cost_weight)

        # record the track power output profile
        if options.hybrid_tracking == False:
            track_gen_pow_sced = thermalBid.extract_pow_s_s(m_track_sced,\
                                                            horizon = options.sced_horizon, \
                                                            verbose = False)
            thermal_track_gen_pow_sced = track_gen_pow_sced
            thermal_generated_sced = track_gen_pow_sced
        else:
            # need to extract P_R and P_T
            # for control power recording and updating the model
            track_gen_pow_sced, \
            thermal_track_gen_pow_sced, \
            thermal_generated_sced = thermalBid.extract_pow_s_s(m_track_sced,\
                                                                horizon = options.sced_horizon,\
                                                                hybrid = True,\
                                                                verbose = False)

        # record the total power delivered and thermal power delivered
        total_power_delivered_arr = simulator.data_manager.extensions['total_power_delivered_arr']
        thermal_power_delivered_arr = simulator.data_manager.extensions['thermal_power_delivered_arr']
        thermal_power_generated_arr = simulator.data_manager.extensions['thermal_power_generated_arr']
        total_power_delivered_arr[h,date_idx] = track_gen_pow_sced[options.bidding_generator][0]
        thermal_power_delivered_arr[h,date_idx] = thermal_track_gen_pow_sced[options.bidding_generator][0]
        thermal_power_generated_arr[h,date_idx] = thermal_generated_sced[options.bidding_generator][0]

        # use the schedule for this step to update the recorder
        sced_tracker_power_record = simulator.data_manager.extensions['sced_tracker_power_record']
        sced_tracker_power_record[options.bidding_generator][:-1] = sced_tracker_power_record[options.bidding_generator][1:]
        sced_tracker_power_record[options.bidding_generator][-1] = thermal_generated_sced[options.bidding_generator][0]

        # update the track model
        thermalBid.update_model_params(m_track_sced,\
                                       sced_tracker_power_record[options.bidding_generator], \
                                       unit = options.bidding_generator,\
                                       hybrid = options.hybrid_tracking)
        thermalBid.reset_constraints(m_track_sced,options.sced_horizon)

prescient.plugins.register_after_operations_callback(after_sced)

def update_observed_thermal_dispatch(options, simulator, ops_stats):

    current_time = simulator.time_manager.current_time
    h = current_time.hour
    date_as_string = current_time.date
    date_idx = simulator.time_manager.dates_to_simulate.index(date_as_string)

    total_power_delivered_arr = simulator.data_manager.extensions['total_power_delivered_arr']

    if options.track_ruc_signal:
        print('Making changes in observed power output using tracking RUC model.')
        g = options.bidding_generator
        ops_stats.observed_thermal_dispatch_levels[g] = total_power_delivered_arr[h,date_idx]

    elif options.track_sced_signal:
        print('Making changes in observed power output using tracking SCED model.')
        g = options.bidding_generator
        ops_stats.observed_thermal_dispatch_levels[g] = total_power_delivered_arr[h,date_idx]

prescient.plugins.register_update_operations_stats_callback(update_observed_thermal_dispatch)

def after_ruc_activation(options, simulator):

    simulator.data_manager.extensions['ruc_dispatch_level_current'] = \
            simulator.data_manager.extensions['ruc_dispatch_level_for_next_period']
    simulator.data_manager.extensions['ruc_dispatch_level_for_next_period'] = None

prescient.plugins.register_after_ruc_activation_callback(after_ruc_activation)
'''
