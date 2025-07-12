#!/usr/bin/env python3
"""
Test database connection to debug connection issues
"""

import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

def test_db_connection():
    """Test database connection with debugging"""
    
    # Get database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5433')),
        'database': os.getenv('DB_NAME', 'amaai_trading'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'P@ssw0rd')
    }
    
    print("Database Configuration:")
    for key, value in db_config.items():
        if key == 'password':
            print(f"  {key}: {'*' * len(str(value))}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTesting database connection...")
    
    try:
        # Test psycopg2 connection
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL connection successful!")
        print(f"   Version: {version[0]}")
        cursor.close()
        conn.close()
        
        # Test SQLAlchemy connection
        try:
            from sqlalchemy import create_engine
            db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            print(f"\nTesting SQLAlchemy URL: postgresql://{db_config['user']}:***@{db_config['host']}:{db_config['port']}/{db_config['database']}")
            
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")
                print("✅ SQLAlchemy connection successful!")
                
        except ImportError:
            print("❌ SQLAlchemy not available")
        except Exception as e:
            print(f"❌ SQLAlchemy connection failed: {e}")
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Check for common issues
        if "could not translate host name" in str(e):
            print("\n🔍 Debugging host name issue:")
            print("   This usually means there's a problem with the host configuration.")
            print("   Check your .env file for correct DB_HOST, DB_USER, and DB_PASSWORD values.")

if __name__ == "__main__":
    test_db_connection()
