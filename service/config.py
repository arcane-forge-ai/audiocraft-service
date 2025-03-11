import json
import os
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment Name
    env: str = "dev"

    # Filestore
    azure_storage_connection_string: str = Field(
        ..., env="AZURE_STORAGE_CONNECTION_STRING")
    azure_storage_account_name: str = Field(...,
                                            env="AZURE_STORAGE_ACCOUNT_NAME")
    azure_container_name: str = Field(..., env="AZURE_CONTAINER_NAME")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
