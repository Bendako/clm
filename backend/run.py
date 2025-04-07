import os
import uvicorn
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Entry point for the application
if __name__ == "__main__":
    logger.info("Starting CLM Backend Service")
    
    # Get configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    # Start the FastAPI application with uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        debug=debug,
        log_level="info" if not debug else "debug"
    )
    
    logger.info("Backend service stopped") 