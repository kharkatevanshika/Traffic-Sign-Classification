import mysql.connector


connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123"
)

cursor = connection.cursor()

# Create database
cursor.execute(
    "CREATE DATABASE IF NOT EXISTS traffic_sign_db"
)

# Select database
cursor.execute(
    "USE traffic_sign_db"
)

# Users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Predictions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    image_name VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
    REFERENCES users(id)
    ON DELETE CASCADE
)
""")

connection.commit()

print("Database and tables created successfully!")

cursor.close()
connection.close()