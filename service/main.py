import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import logging
from routers import info, music_generation

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "colored": {
            "()": "colorlog.ColoredFormatter",
            # Only apply color to the log level
            "format": "%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
        },
    },
    "loggers": {
        "": {  # Root logger
            "level": "INFO",
            "handlers": ["console"],
        },
        "uvicorn": {  # Uvicorn logs
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.error": {  # Uvicorn error logs
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.access": {  # Uvicorn access logs
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}


# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.include_router(info.router, tags=["info"])
app.include_router(music_generation.router, tags=["music_generation"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

