from db_connect import get_connection


try:
    connection = get_connection()

    if connection.is_connected():
        print("MySQL connection successful!")

    connection.close()

except Exception as e:
    print("Database connection failed!")
    print("Error:", e)