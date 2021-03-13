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

opt = Option('--rts_gmlc_data_dir',
             help="the relative path to rts gmlc data set",
             action='store',
             dest='rts_gmlc_data_dir',
             type='str',
             default='./RTS-GMLC/RTS_Data/SourceData/')
prescient.plugins.add_custom_commandline_option(opt)

opt = Option('--price_forecast_dir',
             help="the relative path to price forecasts",
             action='store',
             dest='price_forecast_dir',
             type='str',
             default='../../prescient/plugins/price_forecasts/')
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
    thermal_bid = DAM_thermal_bidding(rts_gmlc_data_dir = options.rts_gmlc_data_dir, \
                                      price_forecast_dir = options.price_forecast_dir,\
                                      generators = [options.bidding_generator],\
                                      horizon = options.ruc_horizon)
    simulator.data_manager.extensions['thermal_bid'] = thermal_bid

    return
prescient.plugins.register_initialization_callback(initialize_bidding_object)

def initialize_tracking_object(options, simulator):

    # initialize the model class
    thermal_track = DAM_thermal_tracking(rts_gmlc_data_dir = options.rts_gmlc_data_dir,\
                                         tracking_horizon = options.sced_horizon,\
                                         generators = [options.bidding_generator])
    simulator.data_manager.extensions['thermal_track'] = thermal_track

    return
prescient.plugins.register_initialization_callback(initialize_tracking_object)

def pass_bid_to_prescient(options, ruc_instance, ruc_date, bids):

    gen_name = options.bidding_generator
    gen_dict = ruc_instance.data['elements']['generator'][gen_name]
    p_cost = [list(bids[t][gen_name].items()) for t in range(options.ruc_horizon)]

    gen_dict['p_cost'] = {'data_type': 'time_series',
                          'values': [{'data_type' : 'cost_curve',
                                     'cost_curve_type':'piecewise',
                                     'values':p_cost[t]} for t in range(options.ruc_horizon)]
                         }

    return

def assemble_project_tracking_signal(options, simulator, hour):

     ## TODO: make this in a for loop (imagine we have a list of bidding generators)
     gen_name = options.bidding_generator

     current_ruc_dispatch = simulator.data_manager.ruc_market_active.thermal_gen_cleared_DA

     market_signals = {gen_name:[]}

     # append corresponding RUC dispatch
     for t in range(hour, hour+options.sced_horizon):
         if t >= 23:
             dispatch = current_ruc_dispatch[(gen_name,23)]
         else:
             dispatch = current_ruc_dispatch[(gen_name,t)]
         market_signals[gen_name].append(dispatch)

     return market_signals

def get_full_projected_trajectory(options, simulator):

    # unpack tracker
    thermal_track = simulator.data_manager.extensions['thermal_track']

    full_projected_trajectory = {}

    for stat in thermal_track.daily_stats:
        full_projected_trajectory[stat] = {}

        ## TODO: we can have a loop here
        gen_name = options.bidding_generator

        # merge the trajectory
        full_projected_trajectory[stat][gen_name] = thermal_track.daily_stats.get(stat)[gen_name] + \
                                                   thermal_track.projection.get(stat)[gen_name]

    thermal_track.clear_projection()

    return full_projected_trajectory

def project_tracking_trajectory(options, simulator, ruc_hour):

    # unpack tracker
    thermal_track = simulator.data_manager.extensions['thermal_track']

    projection_m = thermal_track.clone_tracking_model()

    for hour in range(ruc_hour, 24):

        # assemble market_signals
        market_signals = assemble_project_tracking_signal(options = options, \
                                                          simulator = simulator, \
                                                          hour = hour)
        # solve tracking
        thermal_track.pass_schedule_to_track(m = projection_m,\
                                             market_signals = market_signals, \
                                             last_implemented_time_step = 0,\
                                             hour = hour,\
                                             projection = True)

    return get_full_projected_trajectory(options,simulator)

def bid_into_DAM(options, simulator, ruc_instance, ruc_date, ruc_hour):

    # check if it is first day
    is_first_day = simulator.time_manager.current_time is None

    # unpack bid object
    thermal_bid = simulator.data_manager.extensions['thermal_bid']

    if not is_first_day:

        # solve rolling horizon to get the trajectory
        full_projected_trajectory = project_tracking_trajectory(options, \
                                                                simulator, \
                                                                ruc_hour)
        # update the bidding model
        thermal_bid.update_model(implemented_power_output = full_projected_trajectory['power'],\
                                 implemented_shut_down = full_projected_trajectory['shut_down'], \
                                 implemented_start_up = full_projected_trajectory['start_up'])

    # generate bids
    bids = thermal_bid.stochastic_bidding(ruc_date)
    if is_first_day:
        simulator.data_manager.extensions['current_bids'] = bids
        simulator.data_manager.extensions['next_bids'] = bids
    else:
        simulator.data_manager.extensions['next_bids'] = bids

    # pass to prescient
    pass_bid_to_prescient(options, ruc_instance, ruc_date, bids)

    # record bids
    # thermal_bid.record_bids()

    return
prescient.plugins.register_before_ruc_solve_callback(bid_into_DAM)

def save_ruc_plan(options, simulator, ruc_plan, ruc_date, ruc_hour):

    # save the ruc plan as a property of tracking object

    pass
prescient.plugins.register_after_ruc_generation_callback(save_ruc_plan)

def assemble_sced_tracking_market_signals(options,simulator,sced_instance, hour):

    ## TODO: make this in a for loop (imagine we have a list of bidding generators)
    gen_name = options.bidding_generator

    sced_dispatch = sced_instance.data['elements']['generator'][gen_name]['pg']['values']
    current_ruc_dispatch = simulator.data_manager.ruc_market_active.thermal_gen_cleared_DA
    if simulator.data_manager.ruc_market_pending is not None:
        next_ruc_dispatch = simulator.data_manager.ruc_market_pending.thermal_gen_cleared_DA

    # append the sced dispatch
    market_signals = {gen_name:[sced_dispatch[0]]}

    # append corresponding RUC dispatch
    for t in range(hour+1, hour+options.sced_horizon):
        if t > 23 and simulator.data_manager.ruc_market_pending is not None:
            t = t % 24
            dispatch = next_ruc_dispatch[(gen_name,t)]
        elif t > 23 and simulator.data_manager.ruc_market_pending is None:
            dispatch = sced_dispatch[t-hour]
        else:
            dispatch = current_ruc_dispatch[(gen_name,t)]
        market_signals[gen_name].append(dispatch)

    return market_signals

def track_sced_signal(options, simulator, sced_instance):

    current_date = simulator.time_manager.current_time.date
    current_hour = simulator.time_manager.current_time.hour

    # unpack tracker
    thermal_track = simulator.data_manager.extensions['thermal_track']

    # get market signals
    market_signals = assemble_sced_tracking_market_signals(options = options, \
                                                           simulator = simulator, \
                                                           sced_instance = sced_instance, \
                                                           hour = current_hour)

    # actual tracking
    thermal_track.pass_schedule_to_track(m = thermal_track.model,\
                                         market_signals = market_signals, \
                                         last_implemented_time_step = 0, \
                                         date = current_date,\
                                         hour = current_hour)

    return
prescient.plugins.register_after_operations_callback(track_sced_signal)

def update_observed_thermal_dispatch(options, simulator, ops_stats):

    # unpack tracker
    thermal_track = simulator.data_manager.extensions['thermal_track']
    g = options.bidding_generator
    ops_stats.observed_thermal_dispatch_levels[g] = thermal_track.get_last_delivered_power(generator = g)

    return
prescient.plugins.register_update_operations_stats_callback(update_observed_thermal_dispatch)

def after_ruc_activation(options, simulator):

    # change bids
    current_bids = simulator.data_manager.extensions['next_bids']
    simulator.data_manager.extensions['current_bids'] = current_bids
    simulator.data_manager.extensions['next_bids'] = None
    return
prescient.plugins.register_after_ruc_activation_callback(after_ruc_activation)

def write_customize_results(options, simulator):

    '''
    write results in bidding and tracking objects
    '''
    pass
prescient.plugins.register_after_simulation_callback(write_customize_results)

'''

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
'''
