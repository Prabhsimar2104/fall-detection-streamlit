# FALL-DETECTION/cloudinary_upload.py
# Image upload service for fall detection frames

import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

def upload_fall_image(image_path, user_id):
    """
    Upload fall detection image to Cloudinary
    
    Args:
        image_path: Local path to the image file
        user_id: ID of the elderly user
        
    Returns:
        str: Cloudinary URL of uploaded image, or None if upload failed
    """
    try:
        # Create unique public_id with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        public_id = f"fall_alerts/user_{user_id}/{timestamp}"
        
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            image_path,
            public_id=public_id,
            folder="fall_alerts",
            resource_type="image",
            tags=["fall_detection", f"user_{user_id}"],
            context=f"user_id={user_id}|timestamp={timestamp}",
            overwrite=False
        )
        
        print(f"✅ Image uploaded to Cloudinary: {result['secure_url']}")
        return result['secure_url']
        
    except Exception as e:
        print(f"❌ Cloudinary upload failed: {str(e)}")
        return None

def delete_fall_image(image_url):
    """
    Delete image from Cloudinary (for privacy/cleanup)
    
    Args:
        image_url: Cloudinary URL of the image
        
    Returns:
        bool: True if deleted successfully
    """
    try:
        # Extract public_id from URL
        public_id = image_url.split('/')[-1].split('.')[0]
        
        result = cloudinary.uploader.destroy(f"fall_alerts/{public_id}")
        print(f"✅ Image deleted from Cloudinary: {public_id}")
        return result['result'] == 'ok'
        
    except Exception as e:
        print(f"❌ Cloudinary deletion failed: {str(e)}")
        return False

# Alternative: Use S3 instead of Cloudinary
# Uncomment below for AWS S3 implementation

"""
import boto3
from botocore.exceptions import ClientError

def upload_to_s3(image_path, user_id):
    '''Upload image to AWS S3'''
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        bucket_name = os.getenv('S3_BUCKET_NAME')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"fall-alerts/user_{user_id}/{timestamp}.jpg"
        
        s3_client.upload_file(
            image_path,
            bucket_name,
            key,
            ExtraArgs={
                'ContentType': 'image/jpeg',
                'Metadata': {
                    'user_id': str(user_id),
                    'timestamp': timestamp
                }
            }
        )
        
        url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
        print(f"✅ Image uploaded to S3: {url}")
        return url
        
    except ClientError as e:
        print(f"❌ S3 upload failed: {str(e)}")
        return None
"""