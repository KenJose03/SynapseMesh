import asyncio
import time
from app.core.db import init_db

async def main():
    print("👷 Worker Service is starting...")
    
    # Retry loop to wait for the Database to be ready
    retries = 5
    while retries > 0:
        try:
            await init_db()
            print("✅ Worker successfully connected to Database!")
            break
        except Exception as e:
            retries -= 1
            print(f"⏳ Database not ready yet... retrying in 3s ({retries} retries left)")
            await asyncio.sleep(3)
    
    if retries == 0:
        print("❌ Worker failed to connect to DB after multiple attempts.")
        return

    print("🚀 Worker is online and idling for jobs...")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())