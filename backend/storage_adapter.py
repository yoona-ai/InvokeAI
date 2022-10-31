import os
import boto3
import datetime
import uuid


class S3Adapter:
    def __init__(self):
        self.AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
        self.AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        self.AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')

    def store_from_local_file(self, local_filename, **kwargs):
        dt = datetime.datetime.now()
        s3_filename = f'generated/images/user/{kwargs["user_id"]}/{dt.day}/{dt.month}/{dt.year}/{str(uuid.uuid4())}.png'
        s3_client = boto3.client('s3')

        s3_client.upload_file(local_filename, self.AWS_BUCKET_NAME, s3_filename)
        return s3_filename
