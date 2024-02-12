"""
Yuli Tshuva
"""

import os
from flask import Flask, render_template
import secrets
import shutil
from datetime import datetime, timedelta

random_hex = secrets.token_hex()

app = Flask(__name__, static_folder='static', template_folder="templates")
app.config['UPLOAD_FOLDER'] = os.path.abspath("upload_folder")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=1)
app.config["SECRET_KEY"] = random_hex

os.makedirs("static/temp_files", exist_ok=True)


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
@app.route('/', methods=['GET'])
@app.route('/Home', methods=['GET'])
def home():
    delete_old_files_and_folders()
    return render_template("index.html", active="Home")


@app.route('/Example', methods=['GET'])
def example():
    return render_template("example.html", active="Example")


@app.route('/Help', methods=['GET'])
def help():
    return render_template("help.html", active="Help")


@app.route('/About', methods=['GET'])
def about():
    return render_template("about.html", active="About")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
