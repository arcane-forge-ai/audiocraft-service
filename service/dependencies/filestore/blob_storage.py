from datetime import datetime, timedelta, timezone
from io import BytesIO
import logging
from typing import Union
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
from fastapi import HTTPException, Depends
from config import settings
import os

logger = logging.getLogger(__name__)

try:
    if settings.env.lower() == "local":
        account_url = f"https://{settings.azure_storage_account_name}.blob.core.windows.net"
        default_credential = DefaultAzureCredential()

        blob_service_client = BlobServiceClient(account_url,
                                                credential=default_credential)
    else:
        # TODO: Service Connector is still in preview and doesn't seem to work for our subscription
        # https://learn.microsoft.com/en-us/azure/service-connector/tutorial-python-aks-storage-workload-identity?tabs=azure-portal
        blob_service_client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string)

except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


def upload_to_azure_blob(
    blob_path: str,
    file_or_data: Union[str, BytesIO],
):
    try:
        container_client = blob_service_client.get_container_client(
            settings.azure_container_name)
        blob_client = container_client.get_blob_client(blob_path)

        if isinstance(file_or_data, str):
            # It's a file path
            if not os.path.exists(file_or_data):
                raise HTTPException(status_code=404, detail="File not found")
            with open(file_or_data, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
        elif isinstance(file_or_data, BytesIO):
            # It's a BytesIO object
            blob_client.upload_blob(file_or_data, overwrite=True)
        else:
            raise ValueError(
                "Invalid input type. Expected file path or BytesIO object.")

        return
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_blob_sas_link(blob_path: str):
    try:
        container_client = blob_service_client.get_container_client(
            settings.azure_container_name)
        blob_client = container_client.get_blob_client(blob_path)

        # Check if blob exists
        if not blob_client.exists():
            raise HTTPException(status_code=404,
                                detail=f"File {blob_path} not found")

        # Set SAS expiration time (e.g., 1 hour)
        sas_token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        # Generate the SAS token for the blob
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=settings.azure_container_name,
            blob_name=blob_path,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),  # Grant read access
            expiry=sas_token_expiry)

        # Construct the SAS URL for the user to download
        sas_url = f"{blob_client.url}?{sas_token}"

        return sas_url
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error generating download link: {str(e)}")
