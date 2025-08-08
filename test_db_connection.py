import os
import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error

def test_mysql_connection():
    """Test the MySQL database connection using environment variables."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get database configuration from environment variables
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 3306))
    }
    
    # Print the configuration (without password for security)
    print("Attempting to connect to MySQL database with configuration:")
    print(f"Host: {db_config['host']}")
    print(f"Database: {db_config['database']}")
    print(f"User: {db_config['user']}")
    print(f"Port: {db_config['port']}")
    
    connection = None
    try:
        # Attempt to connect to the database
        connection = mysql.connector.connect(**db_config)
        
        if connection.is_connected():
            # Get server info
            db_info = connection.get_server_info()
            print(f"\n✅ Successfully connected to MySQL Server version {db_info}")
            
            # Get database name
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print(f"Connected to database: {record[0]}")
            
            # Check if tables exist
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            
            if tables:
                print("\nTables in the database:")
                for table in tables:
                    print(f"- {table[0]}")
            else:
                print("\nNo tables found in the database.")
            
            cursor.close()
            
    except Error as e:
        print(f"\n❌ Error connecting to MySQL: {e}")
        
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("\nMySQL connection is closed")

if __name__ == "__main__":
    test_mysql_connection()
