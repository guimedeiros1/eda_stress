# This file implements methods for
# * Analysis of EDA features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import signal_transform as ST
import mycolors as CLRS


# Select features for given subjects
def select_features(df, feature:str, subjects:list=None):
    if subjects is None: # Select all subjects
        return [df[df['subj']==subj][feature].to_numpy() for subj in df['subj'].unique()]
    else:
        return [df[df['subj']==subj][feature].to_numpy() for subj in np.unique(subjects)]

# Tonic analysis
def tonic_feature_anlaysis(df, feature='EDA_Tonic', EDA_SAMPLING_FREQ = 4, subjects:list=None):
    
    # Time reference
    t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_EDA_Tonic = select_features(df, feature, subjects=subjects)

    mean_df = np.mean(all_EDA_Tonic, axis=0)
    med_df = np.median(all_EDA_Tonic, axis=0)
    max_df = np.maximum.reduce(all_EDA_Tonic)
    min_df = np.minimum.reduce(all_EDA_Tonic)

    for eda_tonic in all_EDA_Tonic:plt.plot(t_values/(60*60), eda_tonic,'--', alpha=0.3) 
    plt.plot(t_values/(60*60), max_df, alpha=0.2)
    plt.plot(t_values/(60*60), min_df, alpha=0.2)
    plt.plot(t_values/(60*60), mean_df, label="Mean")
    plt.plot(t_values/(60*60), med_df, label="Median")
    plt.fill_between(t_values/(60*60), min_df, max_df, alpha=0.5)
    plt.title("EDA_Tonic")
    plt.ylabel("µS")
    plt.xlabel("t (hour)")
    plt.legend()
    plt.show()

def tonic_feature_variability_anlaysis(df, t_sec_ref:int, t_sec_step:int, feature='EDA_Tonic', EDA_SAMPLING_FREQ = 4, subjects:list=None, fill_method="confidence", y_label=None, graph_label_suffix="", color=None, show=True, retval=False):
    
    # Time reference
    t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_EDA_Tonic = select_features(df, feature, subjects=subjects)

    # # Feature value means per interval per subject
    # per_subj_means = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+interval_sec) * EDA_SAMPLING_FREQ]) 
    #             for t_sec in range(0, int(t_values[-1]), interval_sec)]
    #             for sub_values in all_EDA_Tonic
    #             ]
    # Feature value means per t_sec_step interval per subject
    per_subj_means = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]
                for sub_values in all_EDA_Tonic
                ]
    
    print(len(per_subj_means))

    # t_n - t_(n-interval)
    per_subj_diff = [ np.diff(sub_values)
                for sub_values in per_subj_means
                ]
    
    # Append last values
    per_subj_diff = [ np.append(sub_values,sub_values[-1])
                for sub_values in per_subj_diff
                ]
    
    mean_df = np.mean(per_subj_diff, axis=0)
    med_df = np.median(per_subj_diff, axis=0)
    max_df = np.maximum.reduce(per_subj_diff)
    min_df = np.minimum.reduce(per_subj_diff)
    std_df = np.std(per_subj_diff, axis=0)
    num_samples = np.shape(per_subj_diff)[0]
    confidence_interval = 1.96 * (std_df / np.sqrt(num_samples))

    def compute_changes(curve):
        ic = dc = 0
        for i in range(1, len(curve)):
            ic += curve[i] > curve[i - 1]
            dc += curve[i] < curve[i - 1]
        tc = ic + dc
        return (ic / tc) * 100, (dc / tc) * 100
    pi, pd = compute_changes(mean_df)
    print("Percentage of times increasing:", pi, color)
    print("Percentage of times decreasing:", pd, color)

    def compute_average_change(curve):
        n = len(curve)
        inc = dec = 0
        for i in range(1, n):
            diff = curve[i] - curve[i-1]
            if diff > 0: inc += abs((diff / curve[i-1]) * 100)
            elif diff < 0: dec += abs((-diff / curve[i-1]) * 100)
        return inc / (n - 1), dec / (n - 1)
    ai, ad = compute_average_change(mean_df)
    print("Average percentage of increase:", ai)
    print("Average percentage of decrease:", ad)

    # x_values = [t_sec/(60*60) for t_sec in range(0, int(t_values[-1]), interval_sec)]
    x_values = [t_sec/(60*60) for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]

    if fill_method == "min_max_median":
        for eda_tonic_diff in per_subj_diff:plt.plot(x_values, eda_tonic_diff,'--', alpha=0.3) 
        plt.plot(x_values, max_df,'+-', label="Maximum"+graph_label_suffix, alpha=0.2, color=color) # plot maximum of all subjects
        plt.plot(x_values, min_df,'_-', label="Minimum"+graph_label_suffix, alpha=0.2, color=color) # plot minimum of all subjects
        plt.plot(x_values, med_df,'*-', label="Median"+graph_label_suffix, color=color)  # plot median of all subjects
        plt.fill_between(x_values, min_df, max_df, alpha=0.5)
    if fill_method == "confidence":
        plt.fill_between(x_values, mean_df - confidence_interval, mean_df + confidence_interval, alpha=0.2)
    
    plt.plot(x_values, mean_df,'*-', label="Mean"+graph_label_suffix, color=color)  # plot mean of all subjects
    plt.plot(x_values, np.ones_like(mean_df)*np.mean(mean_df),'--', label="Overall mean"+graph_label_suffix, color=color)  # plot mean of all subjects
    # plt.title(f"EDA_Tonic_variability per {interval_sec/60} min")
    plt.title(f"EDA_{feature}_variability per {t_sec_step/60} min")
    # plt.xticks(x_values)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xlabel("t (hours)")
    plt.legend()
    if show:
        plt.show()

    if retval:
        return {"x":x_values, "y":mean_df, "conf":confidence_interval}


# Phasic analysis
def phasic_feature_anlaysis_sum(df, feature:str, t_sec_ref:int, t_sec_step:int, EDA_SAMPLING_FREQ = 4, subjects:list=None, fill_method="confidence", y_label=None, graph_label_suffix="", color=None, show=True, retval=False):

    # Time reference
    t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_values = select_features(df, feature, subjects=subjects)

    # Feature value sums per t_sec_step interval per subject
    per_subj_sums = [ [np.sum(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]
                for sub_values in all_values
                ]
    print("len", len(per_subj_sums))

    mean_df = np.mean(per_subj_sums, axis=0)
    med_df = np.median(per_subj_sums, axis=0)
    max_df = np.maximum.reduce(per_subj_sums)
    min_df = np.minimum.reduce(per_subj_sums)
    std_df = np.std(per_subj_sums, axis=0)
    num_samples = np.shape(per_subj_sums)[0]
    confidence_interval = 1.96 * (std_df / np.sqrt(num_samples))
    
    x_values = [t_sec/(60*60) for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]
    if fill_method == "min_max_median":
        for sums in per_subj_sums:plt.plot(x_values, sums,'--', alpha=0.3, color=color) # plot per subject feature sum values
        plt.plot(x_values, max_df,'+-', label="Maximum"+graph_label_suffix, alpha=0.2, color=color) # plot maximum of all subjects
        plt.plot(x_values, min_df,'_-', label="Minimum"+graph_label_suffix, alpha=0.2, color=color) # plot minimum of all subjects
        plt.plot(x_values, med_df,'x-', label="Median"+graph_label_suffix, color=color)  # plot median of all subjects
        plt.fill_between(x_values, min_df, max_df, alpha=0.5)
    if fill_method == "confidence":
        plt.fill_between(x_values, mean_df - confidence_interval, mean_df + confidence_interval, alpha=0.2)
    
    plt.plot(x_values, mean_df,'*-', label="Mean"+graph_label_suffix, color=color)  # plot mean of all subjects
    plt.plot(x_values, np.ones_like(mean_df)*np.mean(mean_df),'--', label="Overall mean"+graph_label_suffix, color=color)  # plot mean of all subjects
    plt.title(f"EDA_{feature}_SUM per {t_sec_step/60} min")
    # plt.xticks(x_values)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xlabel("t (hours)")
    plt.legend()
    if show:
        plt.show()

    
    if retval:
        return {"x":x_values, "y":mean_df, "conf":confidence_interval}


def phasic_feature_anlaysis_mean(df, feature:str, t_sec_ref:int, t_sec_step:int, EDA_SAMPLING_FREQ = 4, subjects:list=None, fill_method="confidence", y_label=None, graph_label_suffix="", color=None, show=True, retval=False):

    # Time reference
    t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_values = select_features(df, feature, subjects=subjects)

    # Feature value sums per t_sec_step interval per subject
    per_subj_sums = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]
                for sub_values in all_values
                ]

    mean_df = np.mean(per_subj_sums, axis=0)
    med_df = np.median(per_subj_sums, axis=0)
    max_df = np.maximum.reduce(per_subj_sums)
    min_df = np.minimum.reduce(per_subj_sums)
    std_df = np.std(per_subj_sums, axis=0)
    num_samples = np.shape(per_subj_sums)[0]
    confidence_interval = 1.96 * (std_df / np.sqrt(num_samples))
    

    x_values = [t_sec/(60*60) for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]
    if fill_method == "min_max_median":
        for sums in per_subj_sums:plt.plot(x_values, sums,'--', alpha=0.3, color=color) # plot per subject feature sum values
        plt.plot(x_values, max_df,'+-', label="Maximum"+graph_label_suffix, alpha=0.2, color=color) # plot maximum of all subjects
        plt.plot(x_values, min_df,'_-', label="Minimum"+graph_label_suffix, alpha=0.2, color=color) # plot minimum of all subjects
        plt.plot(x_values, med_df,'x-', label="Median"+graph_label_suffix, color=color)  # plot median of all subjects
        plt.fill_between(x_values, min_df, max_df, alpha=0.5)
    if fill_method == "confidence":
        plt.fill_between(x_values, mean_df - confidence_interval, mean_df + confidence_interval, alpha=0.2)
    
    plt.plot(x_values, mean_df,'*-', label="Mean"+graph_label_suffix, color=color, alpha=0.6)  # plot mean of all subjects
    plt.plot(x_values, np.ones_like(mean_df)*np.mean(mean_df),'--', label="Overall mean"+graph_label_suffix, color=color)  # plot mean of all subjects
    
    
    plt.title(f"EDA_{feature}_MEAN per {t_sec_step/60} min")
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xlabel("t (hour)")
    # plt.xticks(x_values)
    plt.legend()
    if show:
        plt.show()

    if retval:
        return {"x":x_values, "y":mean_df, "conf":confidence_interval}

def phasic_feature_anlaysis_auc(df, feature:str, t_sec_ref:int, t_sec_step:int, EDA_SAMPLING_FREQ = 4, subjects:list=None, fill_method="confidence", y_label=None, graph_label_suffix="", color=None, show=True, retval=False):

    # Time reference
    t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_values = select_features(df, feature, subjects=subjects)

    # Feature value sums per t_sec_step interval per subject
    per_subj_auc = [ [np.trapz(np.abs(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]), t_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]
                for sub_values in all_values
                ]

    mean_df = np.mean(per_subj_auc, axis=0)
    med_df = np.median(per_subj_auc, axis=0)
    max_df = np.maximum.reduce(per_subj_auc)
    min_df = np.minimum.reduce(per_subj_auc)
    std_df = np.std(per_subj_auc, axis=0)
    num_samples = np.shape(per_subj_auc)[0]
    confidence_interval = 1.96 * (std_df / np.sqrt(num_samples))
    

    x_values = [t_sec/(60*60) for t_sec in range(t_sec_ref, int(t_values[-1]), t_sec_step)]
    if fill_method == "min_max_median":
        for sums in per_subj_auc:plt.plot(x_values, sums,'--', alpha=0.3, color=color) # plot per subject feature sum values
        plt.plot(x_values, max_df,'+-', label="Maximum"+graph_label_suffix, alpha=0.2, color=color) # plot maximum of all subjects
        plt.plot(x_values, min_df,'_-', label="Minimum"+graph_label_suffix, alpha=0.2, color=color) # plot minimum of all subjects
        plt.plot(x_values, med_df,'x-', label="Median"+graph_label_suffix, color=color)  # plot median of all subjects
        plt.fill_between(x_values, min_df, max_df, alpha=0.5)
    if fill_method == "confidence":
        plt.fill_between(x_values, mean_df - confidence_interval, mean_df + confidence_interval, alpha=0.2)
    
    plt.plot(x_values, mean_df,'*-', label="Mean"+graph_label_suffix, color=color)  # plot mean of all subjects
    plt.plot(x_values, np.ones_like(mean_df)*np.mean(mean_df),'--', label="Overall mean"+graph_label_suffix, color=color)  # plot mean of all subjects
    
    
    plt.title(f"EDA_{feature}_MEAN_AUC per {t_sec_step/60} min")
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xlabel("t (hour)")
    # plt.xticks(x_values)
    plt.legend()
    if show:
        plt.show()

    if retval:
        return {"x":x_values, "y":mean_df, "conf":confidence_interval}


# Phasic analysis : SCR Peak Amplitude (y-axis) vs SCR Peak Duration (x-axis)
def phasic_feature_anlaysis_amp_wrt_dur(df, feature_Amp:str, feature_Dur:str, t_sec_ref:int, t_sec_step:int, EDA_SAMPLING_FREQ = 4, subjects:list=None, cmap="Blues", label="", show=True):

    def _mean_non_zero(arr):
        non_zero_values = arr[arr != 0]
        if len(non_zero_values) == 0:
            return 0
        else:
            return np.mean(non_zero_values)


    # Time reference
    # t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_values = select_features(df, feature_Amp, subjects=subjects)

    # Feature value sums per t_sec_step interval per subject
    per_subj_sums_a = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
                for sub_values in all_values
                ]
    # print(len(per_subj_sums_a))

    all_values = select_features(df, feature_Dur, subjects=subjects)
    # Feature value sums per t_sec_step interval per subject
    per_subj_sums_d = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
                for sub_values in all_values
                ]

    # print(f"Nb of sums {per_subj_sums_a}")
    # print(f"Nb of sums d {per_subj_sums_d}")

    clr = CLRS.generate_shades(num_shades=len(subjects)+1,cmap=cmap)
    # for i in range(len(subjects)):
    #     if i == 0:
    #         plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i], label=label)
    #     else :
    #         plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i])

    # Plot mean values
    _ = {}
    for i in range(len(subjects)):
        sd = per_subj_sums_d[i]
        sa = per_subj_sums_a[i]
        
        for idx in range(len(sd)):
            if sd[idx] in _.keys():
                _[sd[idx]] = np.append(_[sd[idx]], [sa[idx]])
            else:
                _[sd[idx]] = [sa[idx]]
    _d=[]
    _a=[]
    for k,v in sorted(_.items()):
        _d = np.append(_d, k)
        _a = np.append(_a, np.mean(v))
    # plt.plot(_d,_a, color=clr[0])
    # # End - Plot mean values

    # std_df = np.std(_a, axis=0)
    # num_samples = np.shape(_a)[0]
    # confidence_interval = 1.96 * (std_df / np.sqrt(num_samples))
    # plt.fill_between(_d, _a - confidence_interval, _a + confidence_interval, alpha=0.2)


    
    # # plt.ylim(top=0.1)
    # # plt.xlim(right=0.08)
    # plt.title(f"EDA {feature_Amp} vs {feature_Dur} (mean) per {t_sec_step/60} min")
    # plt.ylabel("Amplitude (µS)")
    # plt.xlabel("Duration (sec)")
    

    # if show : 
    #     plt.legend()
    #     plt.show()
    return _a, _d
    for i in np.arange(0,np.max(_d), 0.01):
            indices = np.where((_d >= i) & (_d <= i+0.01))[0]
            width = 0.01

            std_df = np.std(_a[indices], axis=0)
            num_samples = np.shape(_a[indices])[0]
            confidence_interval = 1.96 * (std_df / np.sqrt(num_samples))
            
            if show : 
                plt.bar(i,np.mean(_a[indices]), yerr=confidence_interval, width=0.01/3, align="edge", color=clr)
                # print(i)
            else:
                # print(i-width/2)
                plt.bar(i-width/3,np.mean(_a[indices]), yerr=confidence_interval, width=0.01/3, align="edge", color=clr)
        
    if show : 
        plt.ylim(bottom=0)
        plt.xlim(right=max(0.08, np.max(_d)))
        plt.show()
            


# Phasic analysis : SCR Peak Amplitude (y-axis) vs SCR Peak Duration (x-axis)
def phasic_feature_anlaysis_amp_wrt_dur_3D(df, feature_Amp:str, feature_Dur:str, t_sec_ref:int, t_sec_step:int, EDA_SAMPLING_FREQ = 4, subjects:list=None, cmap="Blues", label="", show=True, prev_ax = None):

    # Time reference
    # t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_values = select_features(df, feature_Amp, subjects=subjects)

    # Feature value sums per t_sec_step interval per subject
    per_subj_sums_a = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
                for sub_values in all_values
                ]
    # print(len(per_subj_sums_a))

    all_values = select_features(df, feature_Dur, subjects=subjects)
    # Feature value sums per t_sec_step interval per subject
    per_subj_sums_d = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
                for sub_values in all_values
                ]

    print(f"Nb of sums {np.shape(per_subj_sums_a)} : {per_subj_sums_a}")

    # clr = CLRS.generate_shades(num_shades=len(subjects)+1,cmap=cmap)
    # for i in range(len(subjects)):
    #     if i == 0:
    #         plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i], label=label)
    #     else :
    #         plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i])


    # Plot Amplitude to Duration Ratio of significant peaks
    if len(per_subj_sums_d) == len(per_subj_sums_a):
        if prev_ax is None :
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else : ax = prev_ax

        # print(f"Nb of sums {per_subj_sums_a}")

        per_subj_sums_t = [ [t_sec/60
                for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
                for sub_values in all_values
                ]
        clr = CLRS.generate_shades(num_shades=len(subjects)+1,cmap=cmap)
        for i in range(len(subjects)):
            x_values = per_subj_sums_d[i]
            z_values = per_subj_sums_a[i]
            y_values = per_subj_sums_t[i]
            # print(np.shape(x_values), np.shape(y_values), np.shape(z_values))
            # cmap = plt.cm.viridis

            # Add a dotted line connecting the projection to each point
            zproj = 0
            xproj = 0        
            for j in range(len(z_values)):
                ax.plot([x_values[j], x_values[j]], [y_values[j], y_values[j]], [zproj, z_values[j]], color=clr[i])
                # ax.plot([xproj, x_values[j]], [y_values[j], y_values[j]], [z_values[j], z_values[j]], color='black', alpha=0.1, linestyle='--')


            if i == 0:
                ax.scatter(x_values, y_values, z_values, color=clr[i], label=label)
                # plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i], label=label)
            else :
                ax.scatter(x_values, y_values, z_values, color=clr[i])
                # plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i])

            

        ax.set_ylabel('t min')
        ax.set_zlabel('Amp µS')
        ax.set_xlabel('Dur sec')
        ax.view_init(elev=20, azim=20)

        if show : 
            plt.title(f"EDA {feature_Amp} vs {feature_Dur} (mean) per {t_sec_step/60} min")
            plt.tight_layout()
            plt.legend()
            plt.show()
        else: return ax




# ########        
# # Phasic analysis : SCR Peak Amplitude (y-axis) vs SCR Peak Duration (x-axis)
# def phasic_feature_anlaysis_amp_wrt_dur(df, feature_Amp:str, feature_Dur:str, t_sec_ref:int, t_sec_step:int, EDA_SAMPLING_FREQ = 4, subjects:list=None, cmap="Blues", label="", show=True):

#     # Time reference
#     # t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
#     all_values = select_features(df, feature_Amp, subjects=subjects)

#     # Feature value sums per t_sec_step interval per subject
#     # per_subj_sums_a = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
#     #             for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
#     #             for sub_values in all_values
#     #             ]
#     per_subj_sums_a = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
#                 for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
#                 for sub_values in all_values
#                 ]
    
#     per_subj_sums_a = []
#     for sub_values in all_values:
#         sub_mean = []
#         for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step):
#             sub_values_in_win = sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]
#             sub_values_in_win_nonzero = sub_values_in_win[sub_values_in_win != 0]
#             win_mean = np.mean(sub_values_in_win_nonzero)
#             win_mean = np.nan_to_num(win_mean, nan=0.0)
#             sub_mean = sub_mean+ [win_mean]
#         if len(per_subj_sums_a) > 0:
#             per_subj_sums_a = per_subj_sums_a + [sub_mean]
#         else:
#             per_subj_sums_a = [sub_mean]
    
#     # # print(len(per_subj_sums_a))

#     all_values = select_features(df, feature_Dur, subjects=subjects)
#     # Feature value sums per t_sec_step interval per subject
#     per_subj_sums_d = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
#                 for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
#                 for sub_values in all_values
#                 ]
    
#     per_subj_sums_d = []
#     for sub_values in all_values:
#         sub_mean = []
#         for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step):
#             sub_values_in_win = sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]
#             sub_values_in_win_nonzero = sub_values_in_win[sub_values_in_win != 0]
#             win_mean = np.mean(sub_values_in_win_nonzero)
#             win_mean = np.nan_to_num(win_mean, nan=0.0)
#             sub_mean = sub_mean + [win_mean]
#         if len(per_subj_sums_d) > 0:
#             per_subj_sums_d = per_subj_sums_d + [sub_mean]
#         else:
#             per_subj_sums_d = [sub_mean]

#     print(f"Nb of sums {per_subj_sums_a}")

#     clr = CLRS.generate_shades(num_shades=len(subjects)+1,cmap=cmap)
#     for i in range(len(subjects)):
#         if i == 0:
#             plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i], label=label)
#         else :
#             plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i])

#     # Plot mean values
#     _ = {}
#     for i in range(len(subjects)):
#         sd = per_subj_sums_d[i]
#         sa = per_subj_sums_a[i]
        
#         for idx in range(len(sd)):
#             if sd[idx] in _.keys():
#                 _[sd[idx]] = np.append(_[sd[idx]], [sa[idx]])
#             else:
#                 _[sd[idx]] = [sa[idx]]
#     _d=[]
#     _a=[]
#     for k,v in sorted(_.items()):
#         _d = np.append(_d, k)
#         _a = np.append(_a, np.mean(v))
#     plt.plot(_d,_a, color=clr[0])
#     # End - Plot mean values
    
#     # plt.ylim(top=0.1)
#     plt.xlim(right=0.08*100)
#     plt.title(f"EDA {feature_Amp} vs {feature_Dur} (mean) per {t_sec_step/60} min")
#     plt.ylabel("Amplitude (µS)")
#     plt.xlabel("Duration (sec)")
#     if show : 
#         plt.legend()
#         plt.show()



# Phasic analysis : SCR Peak Amplitude (y-axis) vs SCR Peak Duration (x-axis)
def phasic_feature_anlaysis_amp_wrt_pno(df, feature_Amp:str, feature_Pno:str, t_sec_ref:int, t_sec_step:int, EDA_SAMPLING_FREQ = 4, subjects:list=None, cmap="Blues", label="", show=True):

    # Time reference
    # t_values = df[df['subj']==df['subj'][0]]['t_sec'].values
    
    all_values = select_features(df, feature_Amp, subjects=subjects)

    # Feature value sums per t_sec_step interval per subject
    per_subj_sums_a = [ [np.mean(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
                for sub_values in all_values
                ]
    # print(len(per_subj_sums_a))

    all_values = select_features(df, feature_Pno, subjects=subjects)
    # Feature value sums per t_sec_step interval per subject
    per_subj_sums_d = [ [np.sum(sub_values[t_sec * EDA_SAMPLING_FREQ:-1 + (t_sec+t_sec_step) * EDA_SAMPLING_FREQ]) 
                for t_sec in range(t_sec_ref, int(len(sub_values)/ EDA_SAMPLING_FREQ), t_sec_step)]
                for sub_values in all_values
                ]

    print(f"Nb of sums {per_subj_sums_a}")

    clr = CLRS.generate_shades(num_shades=len(subjects)+1,cmap=cmap)
    for i in range(len(subjects)):
        if i == 0:
            plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i], label=label)
        else :
            plt.plot(per_subj_sums_d[i],per_subj_sums_a[i], 'o', color=clr[i])

    # Plot mean values
    _ = {}
    for i in range(len(subjects)):
        sd = per_subj_sums_d[i]
        sa = per_subj_sums_a[i]
        
        for idx in range(len(sd)):
            if sd[idx] in _.keys():
                _[sd[idx]] = np.append(_[sd[idx]], [sa[idx]])
            else:
                _[sd[idx]] = [sa[idx]]
    _d=[]
    _a=[]
    for k,v in sorted(_.items()):
        _d = np.append(_d, k)
        _a = np.append(_a, np.mean(v))
    plt.plot(_d,_a, color=clr[0])
    # End - Plot mean values
    
    # plt.ylim(top=0.1)
    # plt.xlim(right=0.08)
    plt.title(f"EDA {feature_Amp} (mean) vs {feature_Pno} (sum) per {t_sec_step/60} min")
    plt.ylabel("Amplitude (µS)")
    plt.xlabel("Peak Count (#)")
    
    if show : 
        plt.legend()
        plt.show()


def show_boxplots(_first_vals, _last_vals):

    # Create a figure with two subplots arranged vertically
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 2), sharex=False)
    # fig, axes = plt.subplots(1, len(_box_sets), figsize=(15, 5), sharey=False)
    titles = ['Begining', 'End']
    ylabels = ['Amplitude (µS)','Amplitude (µS)']
    xlabels = ['Dur (sec)','Dur (sec)']
    groups = ["SCR Peak Amplitude vs Duration"]



    _f_l_values = [_first_vals, _last_vals]

    max_x = 0
    for _f_l in _f_l_values:
        for _vals in _f_l:
            _a,_d = _vals    
            max_x = max(max_x, _d.max())

    for fig_idx, _f_l in enumerate(_f_l_values):
        
        width = 0.01

        name_above_below_cutoff = ["High Grades", "Low Grades"]

        for idx, _vals in enumerate(_f_l):
            _a,_d = _vals

            cmap=["Greens", "Reds"]
        
            for i in np.arange(0,np.max(_d), 0.01):
                    
                clr = CLRS.generate_shades(num_shades=2+1,cmap=cmap[idx])
                
                indices = np.where((_d >= i) & (_d <= i+0.01))[0]

                std_df = np.std(_a[indices], axis=0)
                num_samples = np.shape(_a[indices])[0]
                confidence_interval = 1.96 * (std_df / np.sqrt(num_samples))
                
                if idx==1 : 
                    if i==0:
                        axes[fig_idx].bar(i,np.mean(_a[indices]), yerr=confidence_interval, width=width/3, align="edge", color=clr, label=name_above_below_cutoff[idx])
                    else:
                        axes[fig_idx].bar(i,np.mean(_a[indices]), yerr=confidence_interval, width=width/3, align="edge", color=clr)
                    # print(i)
                else:
                    # print(i-width/2)
                    if i==0:
                        axes[fig_idx].bar(i-width/3,np.mean(_a[indices]), yerr=confidence_interval, width=width/3, align="edge", color=clr, label=name_above_below_cutoff[idx])
                    else:
                        axes[fig_idx].bar(i-width/3,np.mean(_a[indices]), yerr=confidence_interval, width=width/3, align="edge", color=clr)

        axes[fig_idx].set_ylim(bottom=0)
        axes[fig_idx].set_xlim(right=0.08 + width/3)
        axes[fig_idx].set_title(titles[fig_idx])
        axes[fig_idx].set_ylabel(ylabels[fig_idx])
        if fig_idx == 1:
            axes[fig_idx].set_xlabel(xlabels[fig_idx])
        axes[fig_idx].legend()
        

    plt.tight_layout()
    plt.plot()    