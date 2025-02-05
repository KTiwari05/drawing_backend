from dotenv import load_dotenv
import os
import redis

load_dotenv()

class ApplicationConfig:
    SECRET_KEY = os.getenv("SECRET_KEY")
    
    # Database configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_DATABASE_URI = r"sqlite:///db.sqlite3"

    # Session configuration (no longer needed with JWT but kept if you have other uses)
    SESSION_TYPE = "redis"
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_REDIS = redis.from_url("redis://localhost:6379")

    # JWT configuration (optional)
    JWT_SECRET_KEY = SECRET_KEY  # Use the same secret key for JWTs
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # Set token expiration time (in seconds, e.g., 1 hour)
