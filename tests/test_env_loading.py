#!/usr/bin/env python3
"""
Environment Variable Loading Test
Tests that .env file is loaded correctly
"""

import os

from dotenv import load_dotenv


def test_env_loading():
    """Test environment variable loading from .env file"""
    print("=" * 60)
    print("Environment Variable Loading Test")
    print("=" * 60)

    # Load .env file
    load_dotenv()

    # Test DB_PROVIDER
    db_provider = os.environ.get("DB_PROVIDER", "NOT SET")
    print(f"✅ DB_PROVIDER: {db_provider}")

    if db_provider == "google_spanner":
        print("✅ Google Spanner provider detected")

        # Test Google Spanner environment variables
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "NOT SET")
        instance = os.environ.get("SPANNER_INSTANCE_ID", "NOT SET")
        database = os.environ.get("SPANNER_DATABASE_ID", "NOT SET")
        credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "NOT SET")

        print(f"✅ GOOGLE_CLOUD_PROJECT: {project}")
        print(f"✅ SPANNER_INSTANCE_ID: {instance}")
        print(f"✅ SPANNER_DATABASE_ID: {database}")
        print(f"✅ GOOGLE_APPLICATION_CREDENTIALS: {credentials}")

        # Check if credentials file exists
        if credentials != "NOT SET" and os.path.exists(credentials):
            print(f"✅ Credentials file exists: {credentials}")
        elif credentials != "NOT SET":
            print(f"⚠️  Credentials file not found: {credentials}")
        else:
            print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")

    # Test Flask configuration
    flask_env = os.environ.get("FLASK_ENV", "NOT SET")
    secret_key = os.environ.get("SECRET_KEY", "NOT SET")
    port = os.environ.get("PORT", "NOT SET")

    print(f"✅ FLASK_ENV: {flask_env}")
    print(f"✅ SECRET_KEY: {'SET' if secret_key != 'NOT SET' else 'NOT SET'}")
    print(f"✅ PORT: {port}")

    print("\n" + "=" * 60)
    print("✅ Environment variable loading test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_env_loading()
