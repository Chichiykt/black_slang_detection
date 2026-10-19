import uvicorn
from app.config import Settings

if __name__ == '__main__':
    settings = Settings()
    # One process owns one reusable model. Reload/multiple workers load extra copies.
    uvicorn.run('app.main:app', host=settings.api_host, port=settings.api_port, workers=1)

