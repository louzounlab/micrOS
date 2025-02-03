"""
Yuli Tshuva
"""
import json
import os
import pandas as pd
from flask import Flask, request, render_template, flash
import traceback
import secrets
from run_models import total
import shutil
import time
from datetime import datetime, timedelta

random_hex = secrets.token_hex()

app = Flask(__name__, static_folder='static', template_folder="templates")
app.config['UPLOAD_FOLDER'] = os.path.abspath("upload_folder")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=1)
app.config["SECRET_KEY"] = random_hex

# Global variable to store the taxa data
taxa_data = None


def load_taxa_data():
    global taxa_data
    file_path = os.path.join(app.static_folder, 'files', 'predicted_next_gen_probiotic_taxa_std_p.csv')
    df = pd.read_csv(file_path)
    # In the Taxa column - remove all text before the "g__":
    df['Taxa'] = "g__" + df['Taxa'].str.split("g__").str[-1]
    taxa_data = df.to_dict(orient='records')
    return taxa_data


def delete_old_files_and_folders():
    directory_path = "static/temp_files"

    if not os.listdir(directory_path):
        return

    # Get the current time
    current_time = datetime.now()

    # Calculate the time threshold (1 hour ago)
    threshold_time = current_time - timedelta(hours=1)

    try:
        # Iterate through files and subdirectories in the directory
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)

                # Check if the file is older than 1 hour
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < threshold_time.timestamp():
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)

                # Check if the directory is older than 1 hour
                if os.path.isdir(dir_path) and os.path.getmtime(dir_path) < threshold_time.timestamp():
                    # Delete the entire directory and its contents
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


@app.route('/impute-form', methods=['POST'])
def impute_form():
    try:
        current_time = time.time()

        folder = os.path.join("static", "temp_files", f"{current_time}")
        os.mkdir(folder)
        tasks = ["miMic", "iMic", "samba", "LOCATE"]
        for task in tasks:
            os.mkdir(os.path.join(folder, task))

        tax_level = request.form.get("tax-level-val")
        normalization = request.form.get("normalization-val").lower()
        reduction = request.form.get("reduction-val")
        tax_group = request.form.get("tax-group-val")
        tax_level_freq = request.form.get("tax-level-freq-val")
        z_score_log = request.form.get("z-score-log-val")
        z_score_rel = request.form.get("z-score-rel-val")
        epsilon = float(request.form.get("epsilon"))
        components = int(request.form.get("components"))

        OTU_file = request.files['OTU-file']
        TAG_file = request.files['TAG-file']
        OTU_file.save(f"static/temp_files/otu{current_time}.csv")
        TAG_file.save(f"static/temp_files/tag{current_time}.csv")
        OTU_file = pd.read_csv(f"static/temp_files/otu{current_time}.csv")
        TAG_file = pd.read_csv(f"static/temp_files/tag{current_time}.csv", index_col=0)
        os.remove(f"static/temp_files/otu{current_time}.csv")
        os.remove(f"static/temp_files/tag{current_time}.csv")

        run_iMic = True if request.form.get("iMic") else False
        run_miMic = True if request.form.get("miMic") else False
        run_samba = True if request.form.get("samba") else False
        run_LOCATE = True if request.form.get("LOCATE") else False

        eval = request.form.get('eval-val')
        sis = request.form.get('sis-val')
        correct_first = True if request.form.get('correct_first-val') == "True" else False
        mimic_pvalue = float(request.form.get('pvalue'))
        mimic_threshold_stat = float(request.form.get('threshold_stat'))

        imic_l1_loss = float(request.form.get("l1_loss"))
        imic_weight_decay = float(request.form.get("weight_decay"))
        imic_lr = float(request.form.get("lr"))
        imic_batch_size = int(request.form.get("batch_size"))
        imic_activation = request.form.get("activation-val")
        imic_dropout = float(request.form.get("dropout"))
        imic_kernel_size_a = int(request.form.get("kernel_size_a"))
        imic_kernel_size_b = int(request.form.get("kernel_size_b"))
        imic_stride = int(request.form.get("stride"))
        imic_padding = int(request.form.get("padding"))
        imic_padding2 = int(request.form.get("padding_2"))
        imic_kernel_size_a_2 = int(request.form.get("kernel_size_a_2"))
        imic_kernel_size_b_2 = int(request.form.get("kernel_size_b_2"))
        imic_stride_2 = int(request.form.get("stride_2"))
        imic_channels = int(request.form.get("channels"))
        imic_channels2 = int(request.form.get("channels_2"))
        imic_linear_dim_divider_1 = int(request.form.get("linear_dim_divider_1"))
        imic_linear_dim_divider_2 = int(request.form.get("linear_dim_divider_2"))
        imic_input_dim = request.form.get("input_dim")

        if imic_input_dim[0] == "(" and imic_input_dim[-1] == ")":
            imic_input_dim = imic_input_dim.split(", ")
            if len(imic_input_dim) == 2 and imic_input_dim[0].isdigit() and imic_input_dim[1].isdigit():
                imic_input_dim = (int(imic_input_dim[0]), int(imic_input_dim[1]))
            else:
                imic_input_dim = (8, 235)
        else:
            imic_input_dim = (8, 235)

        imic_dct = {"l1_loss": imic_l1_loss,
                    "weight_decay": imic_weight_decay,
                    "lr": imic_lr,
                    "batch_size": imic_batch_size,
                    "activation": imic_activation,
                    "dropout": imic_dropout,
                    "kernel_size_a": imic_kernel_size_a,
                    "kernel_size_b": imic_kernel_size_b,
                    "stride": imic_stride,
                    "padding": imic_padding,
                    "padding_2": imic_padding2,
                    "kernel_size_a_2": imic_kernel_size_a_2,
                    "kernel_size_b_2": imic_kernel_size_b_2,
                    "stride_2": imic_stride_2,
                    "channels": imic_channels,
                    "channels_2": imic_channels2,
                    "linear_dim_divider_1": imic_linear_dim_divider_1,
                    "linear_dim_divider_2": imic_linear_dim_divider_2,
                    "input_dim": imic_input_dim}

        if run_LOCATE:
            metab_file = request.files['metab']
            metab_file.save(f"static/temp_files/{current_time}/LOCATE/input.csv")
            metab_df = pd.read_csv(f"static/temp_files/{current_time}/LOCATE/input.csv", index_col=0)
            os.remove(f"static/temp_files/{current_time}/LOCATE/input.csv")
        else:
            metab_df = None

        locate_weight_decay_rep = float(request.form.get("weight_decay_rep"))
        locate_weight_decay_dis = float(request.form.get("weight_decay_dis"))
        locate_lr_rep = float(request.form.get("lr_rep"))
        locate_lr_dis = float(request.form.get("lr_dis"))
        locate_rep_coef = float(request.form.get("rep_coef"))
        locate_dis_coef = float(request.form.get("dis_coef"))
        locate_dropout = float(request.form.get("dropout_"))
        locate_activation_rep = request.form.get("activation_rep-val")
        locate_activation_dis = request.form.get("activation_dis-val")
        locate_neurons = int(request.form.get("neurons"))
        locate_neurons2 = int(request.form.get("neurons2"))
        locate_representation_size = int(request.form.get("representation_size"))
        locate_test_size = float(request.form.get("test_size"))

        samba_cutoff = float(request.form.get("cutoff"))
        samba_metric = request.form.get("metric-val")

        output = total(current_time, df=OTU_file, tag=TAG_file, taxonomy_level=tax_level, taxonomy_group=tax_group,
                       epsilon=epsilon, normalization=normalization, z_scoring=z_score_log,
                       norm_after_rel=z_score_rel, pca=(components, reduction), run_iMic=run_iMic,
                       run_miMic=run_miMic, run_samba=run_samba, run_LOCATE=run_LOCATE, eval=eval, sis=sis,
                       correct_first=correct_first, cutoff=samba_cutoff, metric=samba_metric, iMic_params=imic_dct,
                       metab_df=metab_df, test_size=locate_test_size, representation_size=locate_representation_size,
                       weight_decay_rep=locate_weight_decay_rep, weight_decay_dis=locate_weight_decay_dis,
                       lr_rep=locate_lr_rep, lr_dis=locate_lr_dis, rep_coef=locate_rep_coef, dis_coef=locate_dis_coef,
                       activation_rep=locate_activation_rep, activation_dis=locate_activation_dis,
                       neurons=locate_neurons, neurons2=locate_neurons2, dropout=locate_dropout,
                       mimic_pvalue=mimic_pvalue, mimic_threshold_stat=mimic_threshold_stat)

        render_dict = {}
        if run_iMic:
            render_dict["imic_train"] = os.path.join(folder, "iMic", "train.csv")
            render_dict["imic_test"] = os.path.join(folder, "iMic", "test.csv")
            render_dict["roc_auc_train"], render_dict["roc_auc_test"] = output["roc_auc_train"], output["roc_auc_test"]
        if run_miMic:
            mimic_folder = os.path.join(folder, "miMic")
            render_dict["mimic_paths"] = [os.path.join(mimic_folder, file) for file in os.listdir(mimic_folder)
                                          if file[-3:] in ["png", "svg"]]
            render_dict["mimic_corrs"] = os.path.join(mimic_folder, "df_corrs.csv")
        if run_samba:
            render_dict["samba_png"] = os.path.join(folder, "samba", "umap_plot.png")
            render_dict["samba_csv"] = os.path.join(folder, "samba", "dist_matrix.csv")
        if run_LOCATE:
            render_dict["n_pred"] = output["n_pred"] if run_iMic else output["roc_auc_train"]
            render_dict["locate_path"] = os.path.join(folder, 'LOCATE', "results.csv")
        # Prepare the CSV data for embedding in HTML
        csv_data = output["sd_people"].to_csv(index=True)  # Ensure clean CSV output
        render_dict["sd_people"] = json.dumps(csv_data)  # Safely escape for JS

        return render_template("results_second_edition.html", run_iMic=run_iMic, run_miMic=run_miMic,
                               run_samba=run_samba, run_LOCATE=run_LOCATE, run_SDpeople=True, render_dict=render_dict)

    except Exception as e:
        traceback.print_exc()
        return render_template("error.html", active="", error=str(e))


@app.route('/', methods=['GET'])
@app.route('/Home', methods=['GET'])
def home():
    delete_old_files_and_folders()
    return render_template("index.html", active="Home")


@app.route('/Pngt', methods=['GET'])
def Pngt():
    return render_template("Pngt.html", active="predicted next gen probiotic taxa")


@app.route('/get_taxa_data', methods=['GET'])
def get_taxa_data():
    global taxa_data
    if taxa_data is None:
        taxa_data = load_taxa_data()
    try:
        return {'data': taxa_data}
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}, 500


@app.route('/Example', methods=['GET'])
def example():
    return render_template("example.html", active="Example")


@app.route('/About', methods=['GET'])
def about():
    return render_template("about.html", active="About")


if __name__ == "__main__":
    os.makedirs("static/temp_files", exist_ok=True)
    load_taxa_data()  # Load the taxa data when the application starts
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
