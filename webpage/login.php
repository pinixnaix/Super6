<?php
session_start();

// Dummy username and password for demonstration
$valid_username = "user";
$valid_password = "password";

// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Retrieve username and password from the form
    $username = $_POST["username"];
    $password = $_POST["password"];

    // Validate the username and password
    if ($username === $valid_username && $password === $valid_password) {
        // Authentication successful
        $_SESSION["username"] = $username;
        header("Location: dashboard.php"); // Redirect to dashboard page
        exit;
    } else {
        // Authentication failed
        $error_message = "Invalid username or password.";
    }
}
?>
