-- Create Users Table
CREATE TABLE Users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Emotion Records Table
CREATE TABLE Emotion_Records (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    detected_emotion VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5, 2) NOT NULL,
    ai_response TEXT NOT NULL,
    interaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
);
