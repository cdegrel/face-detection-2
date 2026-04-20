import uvicorn
from src.config import Config

if __name__ == '__main__':
    uvicorn.run(
        'src.server:app',
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG
    )
