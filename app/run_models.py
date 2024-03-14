"""
Yuli Tshuva
Calling the different models.
"""

import os
import MIPMLP
import SAMBA
import samba
from sklearn import metrics
# import LOCATE
from sklearn.model_selection import train_test_split
import pandas as pd
import subprocess
from mimic_da import apply_mimic
from os.path import join


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        return error.decode()
    return output.decode()


def total(current_time, df, tag, taxonomy_level, taxonomy_group, epsilon, normalization, z_scoring, norm_after_rel, pca,
          run_iMic, run_miMic, run_samba, run_LOCATE, eval, sis, correct_first, cutoff, metric, iMic_params, metab_df,
          test_size, representation_size, weight_decay_rep, weight_decay_dis, lr_rep, lr_dis, rep_coef, dis_coef,
          activation_rep, activation_dis, neurons, neurons2, dropout, mimic_pvalue, mimic_threshold_stat):
    if taxonomy_level == "Specie":
        taxonomy_level = 7
    elif taxonomy_level == "Genus":
        taxonomy_level = 6
    elif taxonomy_level == "Family":
        taxonomy_level = 5
    elif taxonomy_level == "Order":
        taxonomy_level = 4

    processed = MIPMLP.preprocess(df=df, tag=tag, taxonomy_level=taxonomy_level, taxnomy_group=taxonomy_group,
                                  epsilon=epsilon, normalization=normalization,
                                  z_scoring=z_scoring, norm_after_rel=norm_after_rel, pca=pca)

    folder = f"static/temp_files/{current_time}"

    samba.micro2matrix(processed, folder, save=True)

    if run_miMic:
        path = join(folder, "miMic")

        taxonomy_selected, samba_output = apply_mimic(folder, tag, eval=eval, threshold_p=mimic_pvalue,
                                                      processed=processed, apply_samba=False, save=True)
        if taxonomy_selected:
            apply_mimic(folder, tag, mode="plot", tax=taxonomy_selected, eval=eval, sis=sis,
                        correct_first=correct_first, samba_output=samba_output, save=True,
                        threshold_p=mimic_pvalue, THRESHOLD_edge=mimic_threshold_stat, path=path)

    if run_samba:
        array_of_imgs, bact_names, ordered_df = samba.micro2matrix(processed, folder, save=False)

        # Calculate the distance matrix according to SAMBA
        dm = samba.build_SAMBA_distance_matrix(folder, imgs=array_of_imgs, ordered_df=ordered_df)

        # dm = samba.build_SAMBA_distance_matrix(folder, metric, cutoff, tag)
        dm.to_csv(os.path.join(folder, 'samba', "dist_matrix.csv"))

        if tag is not None:
            # Saves our plot in folder/umap_plot.png
            SAMBA.plot_umap(dm, tag, os.path.join(folder, 'samba'))

        # Saves into samba folder "umat_plot.png"

    if run_iMic:
        # env1_command = "conda activate oshrit_env && python -c 'from your_module import your_function; your_function()'"
        # output1 = run_command(env1_command)

        dct = MIPMLP.apply_iMic(tag, folder, params=iMic_params)

        dct["y_train"] = list(dct["y_train"])
        dct["y_test"] = list(dct["y_test"])

        fpr_train, tpr_train, thresholds_train = metrics.roc_curve(dct["y_train"], dct["pred_train"])
        roc_auc_train = metrics.auc(fpr_train, tpr_train)

        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(dct["y_test"], dct["pred_test"])
        roc_auc_test = metrics.auc(fpr_test, tpr_test)

        train_path = os.path.join(folder, "iMic", "train.csv")
        test_path = os.path.join(folder, "iMic", "test.csv")

        pd.DataFrame({"pred_train": dct["pred_train"], "y_train": dct["y_train"]}).to_csv(train_path)
        pd.DataFrame({"pred_test": dct["pred_test"], "y_test": dct["y_test"]}).to_csv(test_path)

    if run_LOCATE:
        X_train, X_test, y_train, y_test = train_test_split(processed, metab_df, test_size=test_size)

        model = LOCATE.LOCATE_training(X_train, y_train, X_test, y_test, representation_size, weight_decay_rep,
                                       weight_decay_dis, lr_rep, lr_dis, rep_coef, dis_coef,
                                       activation_rep, activation_dis, neurons, neurons2, dropout)
        # Prediction
        Z_train, n_pred_tr = LOCATE.LOCATE_predict(model, X_train, y_train.columns)
        Z_val, n_pred = LOCATE.LOCATE_predict(model, X_test, y_test.columns)
        Z = pd.concat([Z_val, Z_train])
        save_path = os.path.join(folder, 'LOCATE', "results.csv")
        Z.to_csv(save_path)

    if run_iMic and run_LOCATE:
        return roc_auc_train, roc_auc_test, n_pred
    elif run_iMic:
        return roc_auc_train, roc_auc_test
    elif run_LOCATE:
        return n_pred
    return
