import os 
import boto3 
from pathlib import Path 
from time import gmtime, strftime 
import tempfile
import tarfile
from sagemaker.core.image_uris import get_training_image_uri 
from sagemaker.core.helper.session_helper import Session 


class BaseTraining:
    def __init__(self,payload:dict):
        self.s3_client = boto3.client("s3")
        self.role = payload["role"]
        self.source_folder_path = payload["source_folder_path"]
        self.s3_prefix = payload["s3_prefix"]
        self.s3_bucket = payload["s3_bucket"]
        self.job_name = payload["job_name"]
        self.hyperparameters = payload["hyperparameters"]
        self.train_data_path = payload["train_data_path"]
        self.instance_type = payload["instance_type"]


    
    def upload_source_folder(self):
        job_folder = Path(__file__).parent.parent / self.source_folder_path
        s3_code_prefix = f"{self.s3_prefix}/code"

        if not job_folder.exists():
            raise FileNotFoundError(f"source code folder not found : {self.source_folder_path}")
        
        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                for item in job_folder.iterdir():
                    if item.is_file():
                        tar.add(item, arcname=item.name)
            tmp_path = tmp.name 

        code_key = f"{s3_code_prefix}/sourcedir.tar.gz"
        self.s3_client.upload_file(tmp_path, self.s3_bucket, code_key)
        os.unlink(tmp_path)
        code_s3_uri = f"s3://{self.s3_bucket}/{code_key}"
        return code_s3_uri
    

    def create_training_job(self):
        code_s3_uri = self.upload_source_folder()
        self.hyperparameters["sagemaker_submit_directory"] = code_s3_uri
        sm = boto3.client("sagemaker")
        sess = Session()
        print("instance type", self.instance_type)
        image_uri = get_training_image_uri(
            region=sess.boto_region_name,
            framework="pytorch",
            framework_version="2.8.0",
            py_version="py312",
            instance_type=self.instance_type,
                )
        sm.create_training_job(
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        RoleArn=self.role,
        OutputDataConfig={
            "S3OutputPath": f"s3://{self.s3_bucket}/{self.s3_prefix}/output",
        },
        ResourceConfig={
            "InstanceType": self.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 8,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 7200},
        TrainingJobName=self.job_name,
        HyperParameters=self.hyperparameters,
        Environment={
            "DATABRICKS_HOST": "https://dbc-c8bbff3e-cee5.cloud.databricks.com",
            "DATABRICKS_TOKEN": "dapi90a1c1dcff65efed142f86d6bf2183bc",
        },
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": self.train_data_path.rstrip("/") + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/octet-stream",
                "InputMode": "File",
                "CompressionType": "None",
            },
        ],
    )


if __name__ == "__main__":

    s3_prefix = "pigmenrationrfdetr"
    timestamp = strftime("%Y%m%d-%H%M%S", gmtime())
    s3_prefix = f"{s3_prefix}/{timestamp}"
    job_name = f"rfdetr-pigment{strftime('%Y%m%d-%H%M%S', gmtime())}"
    input_s3_path = "s3://mltrainingodf/input_files/pigment_new/coco_data_split/"
    epochs = 30 
    batch_size = 4
    grad_accum_steps = 4
    lr = 1e-4
    mlflow_tracking_uri = "databricks"
    experiment_name = "/Users/lokeshwaran@opendatafabric.com/pigmentation"
    hyperparameters = {
        "sagemaker_program": "train.py",
        "model-size": "base",
        "epochs": str(epochs),
        "batch-size": str(batch_size),
        "grad-accum-steps": str(grad_accum_steps),
        "lr": str(lr),
        "mlflow-tracking-uri": mlflow_tracking_uri,
        "experiment-name": experiment_name,
    }
    payload = {
        "role" : "arn:aws:iam::654654464550:role/service-role/AmazonSageMakerAdminIAMExecutionRole",
        "source_folder_path" : "rfdetr_train_instance_job", 
        "s3_prefix" : s3_prefix,
        "s3_bucket" : "mltrainingodf",
        "job_name" : job_name,
        "train_data_path" : input_s3_path,
        "instance_type" : "ml.g4dn.xlarge",
        "hyperparameters" : hyperparameters
    }

    base_training = BaseTraining(payload)
    base_training.create_training_job()



