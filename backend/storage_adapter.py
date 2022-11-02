import os
import boto3
import datetime
import uuid
import io
import json


class S3Adapter:
    def __init__(self, environment: str):
        environment_key = environment.upper()
        self.AWS_SECRET_KEY = os.getenv(f'AWS_SECRET_KEY_{environment_key}')
        self.AWS_ACCESS_KEY_ID = os.getenv(f'AWS_ACCESS_KEY_ID_{environment_key}')
        self.AWS_BUCKET_NAME = os.getenv(f'AWS_BUCKET_NAME_{environment_key}')

    def is_valid(self):
        return self.AWS_BUCKET_NAME and self.AWS_BUCKET_NAME and self.AWS_ACCESS_KEY_ID

    def store_from_local_file(self, local_filename, metadata, **kwargs):
        image_uuid = str(uuid.uuid4())
        s3_client = boto3.client('s3', aws_access_key_id=self.AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=self.AWS_SECRET_KEY)
        dt = datetime.datetime.now()

        # TODO JHILL: make one root string for both of these strings since the don't differ much
        s3_image_filename = f'generated/images/user/{kwargs["user_id"]}/{dt.year}/{dt.month}/{dt.day}/{image_uuid}.png'
        s3_client.upload_file(local_filename, self.AWS_BUCKET_NAME, s3_image_filename)

        s3_metadata_filename = f'generated/images/user/{kwargs["user_id"]}/{dt.year}/{dt.month}/{dt.day}/{image_uuid}.json'
        s3_client.upload_fileobj(io.BytesIO(json.dumps(metadata).encode()), self.AWS_BUCKET_NAME, s3_metadata_filename)

        # TODO JHILL: return the metadata filename as well
        return s3_image_filename, s3_metadata_filename
