import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Create the async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create a session factory
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    """Dependency for getting async database sessions."""
    async with async_session() as session:
        yield session

async def init_db():
    """Creates the tables if they don't exist."""
    async with engine.begin() as conn:
        # Import models here to ensure they are registered
        # from app.core.models import JobModel 
        await conn.run_sync(Base.metadata.create_all)