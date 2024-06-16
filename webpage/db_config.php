<?php
// Database credentials
define('DB_SERVER', 'db'); // Use service name defined in docker-compose.yml
define('DB_USERNAME', 'your_username');
define('DB_PASSWORD', 'your_password');
define('DB_NAME', 'your_database_name');

// Attempt to connect to PostgreSQL database
$dsn = "pgsql:host=" . DB_SERVER . ";dbname=" . DB_NAME;
try {
    $pdo = new PDO($dsn, DB_USERNAME, DB_PASSWORD);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("ERROR: Could not connect. " . $e->getMessage());
}
?>
