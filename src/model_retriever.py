import subprocess
import config

# TODO remove this function after the download and upload refactoringg
def execute(cmd):
    print(cmd)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    try:
        output, error = proc.communicate(timeout=1800)
        status = proc.wait()
        print("Status: {}".format(status))
        if status == 0:
            print("Message: {}".format(error.decode("ascii")))
        else:
            error_msg = "Error: {}".format(error.decode("ascii"))
            print(error_msg)
            raise Exception(error_msg)
        print("Return code: {}".format(proc.returncode))
        print(". . . done.")
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        raise Exception(". . . failed.")

# TODO download the model using the cloud storage python client
def download_model(bucket_name, model_dir):
    """
    Function to download the trained model from the GCP bucket
    Args:
        bucket_name (str) - Name of the bucket in GCP
        model_dir (str) - Local model file name.
    """
    dest = str(config.PACKAGE_ROOT)
    cmd = 'gsutil cp -r gs://{bucket_name}/{model_dir} "{dest}"'.format(
            model_dir=model_dir, bucket_name=bucket_name, dest=dest)
    execute(cmd)

# TODO upload the model using the cloud storage python client
def upload_model(model_dir, bucket_name):
    """
    Function to upload the trained model to the GCP bucket
    Args:
        bucket_name (str) - Name of the bucket in GCP
        model_dir (str) - Local model file name.
    """
    cmd = "gsutil cp -r {model_dir} gs://{bucket_name}/".format(
        model_dir=model_dir, bucket_name=bucket_name)
    execute(cmd)
    

    
