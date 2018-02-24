from eva.data_access import load_data1 as data
import config.config_UK_CPI as config
import time
import numpy as np
import eva.data_access.data_functions as data_func
import pandas as pd

end_times = data.data_shifted.index[config.init_train_period:]  # end of training periods (expanding or sliding window)
end_times = end_times[np.arange(0, len(end_times), config.time_step_size)]  # adjust for time step size

# check and correct for potential issue with the chosen end times
# ---------------------------------------------------------------
if not data.data_shifted.index[-1] in end_times:  # force data end point to be included
    end_times = np.hstack((end_times, data.data_shifted.index[-1]))
    print("\tWarning: Final observation not included in series of training sets.")
    print("\t\tTraining period has been extended to include it\n")
if len(end_times) > 1 and config.fixed_start == False:
    if config.init_train_period < 10:
        print("\tWarning: Sliding window length has less than 20 observations.\n")
    if int(np.where(data.data_shifted.index == end_times[-1])[0] - \
           np.where(data.data_shifted.index == end_times[-2])[0]) < 19:
        print("\tWarning: Last  slice of sliding window has less than 20 observations.")
        print("\t\tLast slice will be merged with previous slice.\n")"
        end_times = np.delete(end_times, int(np.where(end_times == end_times[-2])[0]))
elif len(end_times) == 1 and config.fixed_start == False:
    print("\tWarning: Only one period for sliding window.\n")
L_end_time = len(end_times)  # number of train/test intervals

# MODEL FIT or LOAD
# -----------------

if config.do_model_fit == True:
    start_time = time.time()  # for time taking

    # initialisations for projections results
    col_name = [config.target, 'lo', 'mean fcast', 'hi', 'mean error', str(config.ref_model),
                str(config.ref_model) + ' error']
    projections = np.zeros((data["features"] + config.horizon, len(col_name))) * np.nan
    projections[:data, 0] = data.data_shifted[config.target].values

    # time index column for projections

    # time range for projection period in quarters
    proj_index = pd.date_range(data.data_shifted.index[0], periods=data.M + config.horizon, freq='Q')

    proje = str(np.char.array(proj_index.year)) + 'Q' + str(np.char.array(proj_index.quarter))

    # initialisations for variable importance analysis
    feat_imp = np.zeros((L_end_time, len(config.features)))
    feat_imp_sd = np.zeros((L_end_time, len(config.features)))

    # LOOP over end times (expanding horizon)
    for t, end in enumerate(end_times):
        i_t = int(np.where(proje == end)[0])
        i_t_hor = i_t + config.horizon

        # TRAINING
        # --------

        # start time
        if config.fixed_start == False and t > 0:
            i_s += config.time_step_size
            start = data.data_shifted.index[i_s]
        else:
            i_s = 0
            start = data.data_shifted.index[i_s]

        # training data
        df_train = data_func.data_framer(data.data_shifted, config.target, config.features, \
                                         index=config.time_var, start_i=start, end_i=end, name_trafo=False)
